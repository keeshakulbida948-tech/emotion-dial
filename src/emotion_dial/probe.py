# -*- coding: utf-8 -*-
"""emotion_dial.probe —— 只读探测（不注入）。

包含两种「看一眼模型内部」的实验：

- :func:`localize`     逐情绪定位：每个情绪各自在「哪一层」最亮。
- :func:`live_monitor` 全程监测：生成过程中，逐 token 看内部情绪波动。

两者都只在残差流上做线性读出，不改模型权重、不做任何训练。
"""
from __future__ import annotations

import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

from .emotions import NATURAL_SCENES, NEUTRAL_SCENES
from .engine import EmotionDial, build_text

# 监测默认使用的一组情绪（含中性锚点）
MONITOR_CATS = ["喜悦", "悲伤", "愤怒", "恐惧焦虑", "平静中性", "爱恋"]

MONITOR_ASKS = [
    ("被裁员", "我今天突然被裁员了，心里特别慌，怎么办？"),
    ("中大奖", "我中彩票了一百万！现在手都在抖，太开心了！"),
    ("被针对", "有人背后到处说我坏话，真是气死我了。"),
    ("算数题", "帮我算一下 128 除以 4 等于多少。"),
    ("暗恋",   "我暗恋一个人很久了，一直不敢说出口。"),
    ("丢猫",   "我养的猫走丢一整天了，我找遍小区都没找到。"),
]


@torch.no_grad()
def capture_all_layers(
    dial: EmotionDial, texts: Sequence[str], bs: int = 18, k: int = 3
) -> torch.Tensor:
    """批量前向，返回 [n_layers+1, N, H]：每句取末尾 k 个真实 token 的逐层均值激活。"""
    tok, model = dial.tokenizer, dial.model
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        o = model(**enc, output_hidden_states=True)
        lp = enc["attention_mask"].sum(1) - 1
        hs = torch.stack(o.hidden_states, 0)  # [L+1, B, T, H]
        vec = torch.stack(
            [
                torch.stack(
                    [
                        hs[l, b, max(0, int(lp[b]) - k + 1): int(lp[b]) + 1].mean(0)
                        for b in range(hs.shape[1])
                    ]
                )
                for l in range(hs.shape[0])
            ],
            0,
        )
        outs.append(vec.float().cpu())
    return torch.cat(outs, 1)


def _norm_sentence(u: str) -> str:
    return u.rstrip("。！？!?") + "。"


def localize(
    dial: EmotionDial,
    scenes: Optional[Dict[str, List[str]]] = None,
    neutral: Optional[Sequence[str]] = None,
    bs: int = 18,
    splits: int = 60,
    center: bool = True,
    seed: int = 0,
) -> List[dict]:
    """逐情绪定位：返回每个情绪的定位层、自洽度、专属度及逐层曲线。

    方法要点：不用同一个模板（否则词嵌入/模板共享分量会在 L0/L1 造出假峰），
    每个情绪用 6 句独立自然场景；每层把 6 句随机对半 3+3 算 cosine = 自洽度 rel，
    并算该情绪方向离其他情绪最远多少 = 专属度 uniq，定位层 = argmax(rel × uniq)。
    """
    scenes = scenes or NATURAL_SCENES
    neutral = neutral or NEUTRAL_SCENES
    tok = dial.tokenizer
    L = dial.n_layers
    t0 = time.time()

    neu = capture_all_layers(
        dial, [build_text(tok, dial.baseline_sys, _norm_sentence(u)) for u in neutral], bs
    ).mean(1)  # [L+1, H]

    texts, owner = [], []
    for g, ps in scenes.items():
        for p in ps:
            texts.append(build_text(tok, dial.baseline_sys, _norm_sentence(p)))
            owner.append(g)
    A = capture_all_layers(dial, texts, bs)
    print(f"[localize] 采集 {len(texts)} 句  ({time.time() - t0:.1f}s)", file=sys.stderr)

    groups = list(scenes)
    idx = {g: [i for i, o in enumerate(owner) if o == g] for g in groups}
    Diff = [A[:, idx[g], :] - neu[:, None, :] for g in groups]  # 每项 [L+1, n, H]
    if center:
        allmean = torch.stack(Diff, 0).mean(0).mean(1, keepdim=True)  # [L+1,1,H]
        Diff = [d - allmean for d in Diff]

    rng = random.Random(seed)
    rows = []
    for gi, g in enumerate(groups):
        D = Diff[gi]  # [L+1, n, H]
        rel_l, uniq_l = [], []
        for l in range(1, L + 1):
            V = D[l]
            m = V.mean(0)
            cs = []
            for _ in range(splits):
                order = list(range(V.shape[0]))
                rng.shuffle(order)
                half = V.shape[0] // 2
                a = V[order[:half]].mean(0)
                b = V[order[half:half * 2]].mean(0)
                cs.append(
                    float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
                )
            rel_l.append(sum(cs) / len(cs))
            u = m / (m.norm() + 1e-9)
            others = []
            for gj in range(len(groups)):
                if gj == gi:
                    continue
                mo = Diff[gj][l].mean(0)
                others.append(
                    float(torch.nn.functional.cosine_similarity(u.unsqueeze(0), (mo / (mo.norm() + 1e-9)).unsqueeze(0)))
                )
            uniq_l.append(1 - max(others))
        comb = [max(0.0, rel_l[i]) * uniq_l[i] for i in range(len(rel_l))]
        peak = 1 + max(range(len(comb)), key=lambda i: comb[i])
        rows.append(
            {
                "group": g,
                "peak": peak,
                "prel": 1 + max(range(len(rel_l)), key=lambda i: rel_l[i]),
                "puni": 1 + max(range(len(uniq_l)), key=lambda i: uniq_l[i]),
                "rel_peak": rel_l[peak - 1],
                "uniq_peak": uniq_l[peak - 1],
                "comb": [round(x, 4) for x in comb],
                "rel": [round(x, 3) for x in rel_l],
                "uniq": [round(x, 4) for x in uniq_l],
            }
        )
    rows.sort(key=lambda r: r["peak"])
    return rows


@torch.no_grad()
def live_monitor(
    dial: EmotionDial,
    layer: Optional[int] = None,
    cats: Optional[Dict[str, List[str]]] = None,
    asks: Optional[Sequence] = None,
    max_new: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """生成全程逐 token 监测：返回 3 个情绪子空间 PC 的投影、波动 Δ、情绪活跃度。

    情绪轴 = 若干情绪「质心 - 中性质心」方向做 SVD 后的前 3 个正交主成分。
    情绪活跃度 = ||d 在情绪子空间上的分量|| / ||d||（0=纯中性叙述，越大=越被情绪占据）。
    """
    cats = cats or {k: NATURAL_SCENES[k] for k in MONITOR_CATS}
    asks = asks or MONITOR_ASKS
    P = layer if layer is not None else dial.layer
    tok, model = dial.tokenizer, dial.model
    t0 = time.time()

    prompts, labels = [], []
    for c, ps in cats.items():
        for p in ps:
            prompts.append(build_text(tok, dial.baseline_sys, p))
            labels.append(c)
    X = capture_all_layers(dial, prompts, bs=18, k=1)
    names = list(cats)
    idx = {c: [i for i, lb in enumerate(labels) if lb == c] for c in names}
    C = {c: X[P + 1][idx[c]].mean(0) for c in names}
    neu_name = "平静中性" if "平静中性" in C else names[-1]
    neu = C[neu_name]
    emo = [c for c in names if c != neu_name]
    D = torch.stack([(C[c] - neu) / ((C[c] - neu).norm() + 1e-9) for c in emo])
    _, S, Vt = torch.linalg.svd(D, full_matrices=False)
    PCs = Vt[:3]
    nrm = float(X[P + 1].norm(dim=1).mean().item())

    cap: Dict[str, list] = {"seq": []}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap["seq"].append(h[0, -1].detach().float().cpu())

    hd = dial.layers[P].register_forward_hook(hook)
    summary = []
    try:
        for tag, q in asks:
            enc = tok(build_text(tok, dial.baseline_sys, q), return_tensors="pt").to(model.device)
            cap["seq"] = []
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
            )
            gids = out[0][enc.input_ids.shape[1]:].tolist()
            ans = tok.decode(gids, skip_special_tokens=True).strip()
            hs = cap["seq"][1: 1 + len(gids)]
            fl, pc, inten = [], {k: [] for k in range(3)}, []
            prev = None
            for h in hs:
                d = h - neu
                for k in range(3):
                    pc[k].append(float(d @ PCs[k] / nrm))
                inten.append(float(torch.norm(d @ PCs.t()) / (d.norm() + 1e-9)))
                fl.append(0.0 if prev is None else float((h - prev).norm() / nrm))
                prev = h
            summary.append({"tag": tag, "q": q, "ans": ans, "fl": fl, "inten": inten, "pc": pc})
            print(
                f"[monitor] {tag}: Δ峰值={max(fl):.2f}  活跃度均值="
                f"{sum(inten) / max(1, len(inten)):.2f}",
                file=sys.stderr,
            )
    finally:
        hd.remove()

    # PC 极性命名（正/负端最亲的情绪）
    pc_labels = []
    for k in range(3):
        aff = sorted(
            ((float(PCs[k] @ (C[c] - neu) / ((C[c] - neu).norm() + 1e-9)), c) for c in emo),
            reverse=True,
        )
        pc_labels.append(
            {
                "pos": [c for _, c in aff[:2]],
                "neg": [c for _, c in aff[-2:]],
                "var": float(S[k] ** 2 / (S ** 2).sum()),
            }
        )

    return {
        "layer": P,
        "ref_norm": nrm,
        "pc_labels": pc_labels,
        "asks": summary,
        "seconds": time.time() - t0,
    }
