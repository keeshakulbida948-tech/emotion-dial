# -*- coding: utf-8 -*-
"""示例 2：逐情绪定位——每个情绪自己在哪一层最亮。

    python examples/02_layer_localize.py --model Qwen/Qwen3-0.6B --out localize.png
"""
import argparse
import json

from emotion_dial import EmotionDial
from emotion_dial.probe import localize
from emotion_dial.viz import plot_localize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", default="localize.png")
    ap.add_argument("--json", default="localize.json")
    args = ap.parse_args()

    dial = EmotionDial(args.model, layer=0, load_4bit=args.load_4bit)
    rows = localize(dial)

    for r in rows:
        print(f"{r['group']:<12} 定位层 L*={r['peak']:<3} 自洽={r['rel_peak']:.2f} 专属={r['uniq_peak']:.2f}")

    plot_localize(rows, dial.n_layers, args.out)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out} / {args.json}")


if __name__ == "__main__":
    main()
