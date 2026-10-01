# -*- coding: utf-8 -*-
"""示例 3：逐 token 监测——问问题时，模型内部情绪在哪一步、波动多大。

    python examples/03_live_monitor.py --model Qwen/Qwen3-0.6B --layer 16 --out live.png
"""
import argparse

from emotion_dial import EmotionDial
from emotion_dial.probe import live_monitor
from emotion_dial.viz import plot_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--out", default="live.png")
    args = ap.parse_args()

    dial = EmotionDial(args.model, layer=args.layer, load_4bit=args.load_4bit)
    res = live_monitor(dial, layer=args.layer, max_new=args.max_new)

    print("情绪轴:", " | ".join(f"PC{k+1}: +{'/'.join(lb['pos'])} −{'/'.join(lb['neg'])}" for k, lb in enumerate(res["pc_labels"])))
    for s in res["asks"]:
        inten = s["inten"]
        print(f"{s['tag']:<5} 活跃度 均={sum(inten)/max(1,len(inten)):.2f} 峰={max(inten):.2f}")

    plot_live(res, args.out, model_name=args.model)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
