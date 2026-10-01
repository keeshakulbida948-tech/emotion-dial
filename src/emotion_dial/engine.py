# -*- coding: utf-8 -*-
"""emotion_dial.engine —— 核心引擎。

不重训、不重载模型，随时改 alpha 即可实时改变模型「情绪」。

抽取：每个情绪 E，用「系统提示: 我此刻感到 E」对比「中性系统提示」，取目标层末位
      token 的残差流激活做差 => 原始情绪方向 v_E（未归一化）。
注入：在目标层挂 forward hook，每次前向把 sum_i alpha_i * v_i 加到残差流。
      alpha=1 约等于「一整份情绪差量」，可在生成之间随时改，甚至逐 token 改。

面向 Qwen3 / Qwen3.5(混合线性注意力) 及通用 HF 因果模型。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Sequence

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

try:  # 让 Windows 控制台也能安全打印中文
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .emotions import BASELINE_SYS, DEFAULT_PROBES, EMOTION_PROMPTS


def build_text(tokenizer, sys_prompt: str, user: str) -> str:
    """按模型的 chat 模板拼一段文本（关闭 thinking，便于取干净的激活）。"""
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def find_decoder_layers(model):
    """找到 decoder layer 的 ModuleList（兼容 Qwen3 / Qwen3.5 及多模态包装）。"""
    best = (None, None)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 6:
            kids = list(mod.children())[:4]
            if kids and all(
                ("DecoderLayer" in type(k).__name__) or hasattr(k, "mlp") for k in kids
            ):
                if best[1] is None or len(mod) > len(best[1]):
                    best = (name, mod)
    return best


class EmotionDial:
    """一个模型的「情绪旋钮」。"""

    def __init__(
        self,
        model_path: str,
        layer: int = 16,
        dtype: str = "bfloat16",
        device: str = "cuda",
        device_map=None,
        max_memory=None,
        load_4bit: bool = False,
        emotions: Optional[Dict[str, str]] = None,
        probes: Optional[Sequence[str]] = None,
        baseline_sys: str = BASELINE_SYS,
    ):
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        td = getattr(torch, dtype)
        kw = dict(trust_remote_code=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kw["dtype"] = td
        kw["device_map"] = device if device_map is None else device_map
        if max_memory is not None:
            kw["max_memory"] = max_memory

        self.model = None
        last = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                if cls_name == "AutoModelForCausalLM":
                    cls = AutoModelForCausalLM
                else:
                    import transformers

                    cls = transformers.AutoModelForImageTextToText
                self.model = cls.from_pretrained(model_path, **kw)
                break
            except Exception as e:  # 依次尝试不同加载类
                last = e
        if self.model is None:
            raise RuntimeError(f"加载失败: {last}")
        self.model.eval()

        self.layer_name, self.layers = find_decoder_layers(self.model)
        if self.layers is None:
            raise RuntimeError("找不到 decoder layers")
        self.n_layers = len(self.layers)
        if layer < 0:
            layer = self.n_layers + layer
        self.layer = layer

        self.emotions = dict(emotions or EMOTION_PROMPTS)
        self.probes = list(probes or DEFAULT_PROBES)
        self.baseline_sys = baseline_sys

        self.directions: Dict[str, torch.Tensor] = {}
        self.ref_norm: float = 1.0
        self.alphas: Dict[str, float] = {e: 0.0 for e in self.emotions}
        self._hook = None
        self._cap: Dict[str, torch.Tensor] = {}

        print(
            f"[load] {type(self.model).__name__}  layers={self.n_layers} @ {self.layer_name}"
            f"  dtype={dtype}  inject_layer={self.layer}  ({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )

    # ---------------------------------------------------------------- hooks --
    def _mod(self):
        return self.layers[self.layer]

    @torch.no_grad()
    def _capture_last(self, texts: Sequence[str], k: int = 1) -> torch.Tensor:
        """对每条文本，取目标层末位（或末尾 k 个真实 token 的均值）激活。"""
        cap: Dict[str, torch.Tensor] = {}

        def h(mod, inp, out):
            hh = out[0] if isinstance(out, tuple) else out
            cap["v"] = hh[0, -k:].detach().float().mean(0).cpu()

        hd = self._mod().register_forward_hook(h)
        try:
            for t in texts:
                ids = self.tokenizer(t, return_tensors="pt").to(self.model.device)
                self.model(**ids)
                cap.setdefault("all", [])
                cap["all"] = cap["all"] + [cap["v"]]
        finally:
            hd.remove()
        return torch.stack(cap["all"])

    # -------------------------------------------------------------- extract --
    def extract(self) -> Dict[str, torch.Tensor]:
        """抽取所有情绪方向，返回 {emotion: vector}。"""
        print("[extract] 抽取情绪方向 ...", file=sys.stderr)
        base = self._capture_last(
            [build_text(self.tokenizer, self.baseline_sys, u) for u in self.probes]
        ).mean(0)
        self.ref_norm = float(base.norm().item())
        for e, sp in self.emotions.items():
            acts = self._capture_last(
                [build_text(self.tokenizer, sp, u) for u in self.probes]
            ).mean(0)
            d = acts - base
            self.directions[e] = d
            print(f"         {e:10s} ||v||={d.norm().item():6.3f}", file=sys.stderr)
        print(
            f"[extract] 残差流基准范数 ||h||≈{self.ref_norm:.2f}  "
            f"(alpha=1 即注入一整份情绪差量)",
            file=sys.stderr,
        )
        return self.directions

    # ------------------------------------------------------------ 持久化 --
    def save_vectors(self, path: str):
        torch.save(
            {
                "directions": {e: v.clone() for e, v in self.directions.items()},
                "ref_norm": self.ref_norm,
                "layer": self.layer,
                "n_layers": self.n_layers,
            },
            path,
        )
        print(f"[saved] 情绪方向 -> {path}", file=sys.stderr)

    def load_vectors(self, path: str):
        blob = torch.load(path, map_location="cpu")
        self.directions = blob["directions"]
        self.ref_norm = float(blob.get("ref_norm", 1.0))
        self.layer = int(blob.get("layer", self.layer))
        for e in self.directions:
            self.alphas.setdefault(e, 0.0)
        print(f"[load] 情绪方向 <- {path}  ({len(self.directions)} 个)", file=sys.stderr)
        return self.directions

    # ---------------------------------------------------------------- steer --
    def attach(self):
        """在目标层挂上注入 hook（幂等）。"""
        if self._hook is not None:
            return self
        dial = self

        def hook(mod, inp, out):
            is_t = isinstance(out, tuple)
            h = out[0] if is_t else out
            add = None
            for e, a in dial.alphas.items():
                if a and e in dial.directions:
                    dv = dial.directions[e].to(device=h.device, dtype=h.dtype)
                    add = (dv * a) if add is None else add + dv * a
            if add is None:
                return out
            h = h + add
            return (h,) + out[1:] if is_t else h

        self._hook = self._mod().register_forward_hook(hook)
        return self

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def set(self, **alphas: float):
        """设置若干情绪的 alpha，例如 dial.set(happy=1.0, sad=0.3)。"""
        for e, a in alphas.items():
            if e not in self.alphas:
                raise KeyError(f"未知情绪: {e}（可用: {', '.join(self.alphas)}）")
            self.alphas[e] = float(a)
        return self

    def reset(self):
        for e in self.alphas:
            self.alphas[e] = 0.0
        return self

    # -------------------------------------------------------------- generate --
    @torch.no_grad()
    def respond(
        self,
        user: str,
        max_new: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = 0,
        system: Optional[str] = None,
    ) -> str:
        text = build_text(self.tokenizer, system or self.baseline_sys, user)
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0][ids.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False
