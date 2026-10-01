# -*- coding: utf-8 -*-
"""emotion-dial —— 找到、读取并拨动语言模型内部的「情绪方向」。

用法速览::

    from emotion_dial import EmotionDial, find_decoder_layers
    from emotion_dial.probe import localize, live_monitor

    dial = EmotionDial("Qwen/Qwen3-0.6B", layer=16)
    dial.extract()                      # 抽取情绪方向
    with dial:                          # 自动 attach / detach
        dial.set(happy=1.0, sad=-0.5)
        print(dial.respond("你好呀"))

    rows = localize(dial)               # 逐情绪定位层
    res  = live_monitor(dial, layer=16) # 逐 token 波动监测
"""
from .engine import EmotionDial, build_text, find_decoder_layers
from .emotions import (
    BASELINE_SYS,
    DEFAULT_PROBES,
    EMOTION_PROMPTS,
    NATURAL_SCENES,
    NEUTRAL_SCENES,
)

__version__ = "0.1.0"

__all__ = [
    "EmotionDial",
    "build_text",
    "find_decoder_layers",
    "EMOTION_PROMPTS",
    "BASELINE_SYS",
    "DEFAULT_PROBES",
    "NATURAL_SCENES",
    "NEUTRAL_SCENES",
]
