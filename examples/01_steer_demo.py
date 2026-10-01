# -*- coding: utf-8 -*-
"""示例 1：抽取情绪方向，并对比不同 alpha 下的生成。

    python examples/01_steer_demo.py --model Qwen/Qwen3-0.6B --layer 16
"""
import argparse

from emotion_dial import EmotionDial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--prompt", default="你好！跟我说一句鼓励的话。")
    args = ap.parse_args()

    dial = EmotionDial(args.model, layer=args.layer, load_4bit=args.load_4bit)
    dial.extract()

    for name, alphas in [
        ("neutral", {}),
        ("happy +1", {"happy": 1.0}),
        ("happy +2", {"happy": 2.0}),
        ("sad +2", {"sad": 2.0}),
        ("angry +2", {"angry": 2.0}),
        ("happy+2 / sad-2", {"happy": 2.0, "sad": -2.0}),
    ]:
        dial.reset().set(**alphas)
        print(f"\n=== {name} ===")
        print(dial.respond(args.prompt, max_new=80))
    dial.detach()


if __name__ == "__main__":
    main()
