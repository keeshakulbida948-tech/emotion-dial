# -*- coding: utf-8 -*-
"""emotion-dial 命令行工具。

    emotion-dial extract  --model Qwen/Qwen3-0.6B --layer 16 --save vecs.pt
    emotion-dial demo     --model Qwen/Qwen3-0.6B --layer 16 --prompt "跟我说说今天"
    emotion-dial repl     --model Qwen/Qwen3-0.6B --layer 16
    emotion-dial localize --model Qwen/Qwen3.5-4B --load-4bit --out localize.png
    emotion-dial monitor  --model Qwen/Qwen3.5-4B --layer 21 --load-4bit --out live.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _add_common(ap):
    ap.add_argument("--model", required=True, help="HF 模型名或本地路径")
    ap.add_argument("--layer", type=int, default=16, help="注入/探测目标层（负数为倒数）")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", action="store_true", help="4bit 量化加载（省显存）")
    ap.add_argument("--load-vectors", default=None, help="从文件读取已抽取的情绪方向")


def _build(args, layer=None):
    from .engine import EmotionDial

    return EmotionDial(
        args.model,
        layer=args.layer if layer is None else layer,
        dtype=args.dtype,
        device=args.device,
        load_4bit=args.load_4bit,
    )


def cmd_extract(args):
    dial = _build(args)
    dial.extract()
    if args.save:
        dial.save_vectors(args.save)
    print("\n情绪方向（||v||）:")
    for e, v in dial.directions.items():
        print(f"  {e:10s} {v.norm().item():8.3f}")
    return 0


def cmd_demo(args):
    dial = _build(args)
    if args.load_vectors:
        dial.load_vectors(args.load_vectors)
    else:
        dial.extract()
    user = args.prompt or "我最近有点提不起劲，跟我说点什么。"
    settings = [
        ("neutral", {}),
        ("happy +1.0", {"happy": 1.0}),
        ("happy +2.0", {"happy": 2.0}),
        ("sad +2.0", {"sad": 2.0}),
        ("angry +2.0", {"angry": 2.0}),
        ("calm +1.5", {"calm": 1.5}),
        ("happy+2 / sad-2", {"happy": 2.0, "sad": -2.0}),
    ]
    print("\n" + "=" * 74)
    print(f"用户输入：{user}")
    print("=" * 74)
    with dial:
        for name, alphas in settings:
            dial.reset().set(**alphas)
            out = dial.respond(user, max_new=args.max_new)
            print(f"\n【{name}】")
            print("  " + out.replace("\n", "\n  "))
    print("\n" + "=" * 74)
    return 0


def cmd_repl(args):
    dial = _build(args)
    if args.load_vectors:
        dial.load_vectors(args.load_vectors)
    else:
        dial.extract()
    print("\n=== 情绪旋钮 REPL ===")
    print("可用: " + ", ".join(dial.alphas))
    print("输入示例:  happy 1.5 | happy 1 sad -0.5 | 0(清零) | @prompt 新用户输入 | 直接打字=当作用户输入 | quit\n")
    user = args.prompt or "我最近有点提不起劲，跟我说点什么。"
    with dial:
        while True:
            active = {k: v for k, v in dial.alphas.items() if v}
            try:
                line = input(f"[user={user}] knobs={active} > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("quit", "exit", "q"):
                break
            if line == "0":
                dial.reset()
                continue
            if line.startswith("@prompt"):
                user = line[7:].strip() or user
                continue
            if line:
                toks = line.replace(",", " ").split()
                if len(toks) % 2 == 0 and all(t in dial.alphas for t in toks[0::2]):
                    for e, v in zip(toks[0::2], toks[1::2]):
                        try:
                            dial.alphas[e] = float(v)
                        except ValueError:
                            print("  值必须是数字")
                    continue
                user = line
            print("\n  " + dial.respond(user, max_new=args.max_new).replace("\n", "\n  ") + "\n")
    return 0


def cmd_localize(args):
    from .probe import localize

    dial = _build(args)
    rows = localize(dial, bs=args.bs, splits=args.splits, center=not args.no_center)
    L = dial.n_layers
    print("=" * 84)
    print(f"{'情绪':<12}{'定位层L*':>9}{'自洽度':>8}{'专属度':>8}{'自洽峰':>8}{'专属峰':>8}")
    print("=" * 84)
    for r in rows:
        print(
            f"{r['group']:<12}{r['peak']:>9}{r['rel_peak']:>8.2f}{r['uniq_peak']:>8.2f}"
            f"{r['prel']:>8}{r['puni']:>8}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.json}")
    if args.out:
        from .viz import plot_localize

        plot_localize(rows, L, args.out)
        print(f"[saved] {args.out}")
    return 0


def cmd_monitor(args):
    from .probe import live_monitor

    dial = _build(args)
    res = live_monitor(dial, layer=args.layer, max_new=args.max_new)
    for s in res["asks"]:
        inten = s["inten"]
        print(
            f"\n### 【{s['tag']}】{s['q']}\n    答: {s['ans'][:120]}\n"
            f"    Δ峰值={max(s['fl']):.2f}  活跃度 均={sum(inten)/max(1,len(inten)):.2f} 峰={max(inten):.2f}"
        )
    if args.out:
        from .viz import plot_live

        plot_live(res, args.out, model_name=os.path.basename(args.model.rstrip("/\\")))
        print(f"\n[saved] {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        print(f"[saved] {args.json}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emotion-dial", description="情绪旋钮：找到、读取并拨动 LLM 内部的情绪方向")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("extract", help="抽取情绪方向")
    _add_common(a)
    a.add_argument("--save", default=None, help="保存方向到 .pt")
    a.set_defaults(func=cmd_extract)

    a = sub.add_parser("demo", help="对比不同 alpha 下的生成")
    _add_common(a)
    a.add_argument("--prompt", default=None)
    a.add_argument("--max-new", type=int, default=120)
    a.set_defaults(func=cmd_demo)

    a = sub.add_parser("repl", help="交互式调情绪")
    _add_common(a)
    a.add_argument("--prompt", default=None)
    a.add_argument("--max-new", type=int, default=120)
    a.set_defaults(func=cmd_repl)

    a = sub.add_parser("localize", help="逐情绪定位层（只读）")
    _add_common(a)
    a.add_argument("--bs", type=int, default=18)
    a.add_argument("--splits", type=int, default=60)
    a.add_argument("--no-center", action="store_true")
    a.add_argument("--out", default=None, help="保存热力图 png")
    a.add_argument("--json", default=None, help="保存原始结果 json")
    a.set_defaults(func=cmd_localize)

    a = sub.add_parser("monitor", help="逐 token 内部波动监测（只读）")
    _add_common(a)
    a.add_argument("--max-new", type=int, default=80)
    a.add_argument("--out", default=None, help="保存曲线 png")
    a.add_argument("--json", default=None, help="保存原始结果 json")
    a.set_defaults(func=cmd_monitor)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
