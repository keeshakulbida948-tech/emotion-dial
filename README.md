# emotion-dial · 情绪旋钮

> 找到、读取并**拨动**大语言模型内部的「情绪方向」——不训练、不重载模型，实时改一个数就能改气质。

`emotion-dial` 是一个小巧的可解释性 / 激活操控工具包。它在残差流里定位「情绪方向」，
既能**只读地看一眼**模型内部在想什么（逐层定位、逐 token 波动），
也能**注入一个向量**把模型推向某种情绪。面向 Qwen3 / Qwen3.5 及通用 HuggingFace 因果模型。

[English README →](README_EN.md)

---

## 亮点

- **零训练**：全靠「对比系统提示 → 取激活差 → 挂 hook 注入」，几分钟上手。
- **情绪手术台**：
  - `extract` 抽 8 个情绪的原始方向向量（可保存复用）；
  - `repl` 交互式调 alpha，像拧旋钮一样调情绪；
  - `localize` 逐情绪定位「它自己在哪一层最亮」；
  - `monitor` 生成全程逐 token 监测内部情绪波动。
- **自带可视化**：定位热力图、逐 token 三联曲线图，开箱即出 png。
- **模型无关**：自动找 decoder layers，兼容 Qwen3 / Qwen3.5 / 多模态包装；支持 4bit 量化加载。

## 安装

```bash
git clone https://github.com/<you>/emotion-dial.git
cd emotion-dial
pip install -e .            # 或: pip install -e ".[viz]"  含 matplotlib
```

依赖：`torch`、`transformers`、`numpy`（可视化需 `matplotlib`，量化需 `bitsandbytes`）。

## 快速开始（命令行）

```bash
# 1) 抽情绪方向，保存复用
emotion-dial extract --model Qwen/Qwen3-0.6B --layer 16 --save vecs.pt

# 2) 对比不同情绪下的生成
emotion-dial demo --model Qwen/Qwen3-0.6B --layer 16 --load-vectors vecs.pt \
                  --prompt "你好，跟我说一句鼓励的话。"

# 3) 交互式调情绪（输入 `happy 1.5`、`happy 1 sad -0.5`、`0` 清零、`quit` 退出）
emotion-dial repl --model Qwen/Qwen3-0.6B --layer 16

# 4) 只读：逐情绪定位层（出热力图）
emotion-dial localize --model Qwen/Qwen3.5-4B --load-4bit --out localize.png --json localize.json

# 5) 只读：逐 token 内部波动监测（出曲线图）
emotion-dial monitor --model Qwen/Qwen3.5-4B --layer 21 --load-4bit --out live.png
```

## 快速开始（Python）

```python
from emotion_dial import EmotionDial
from emotion_dial.probe import localize, live_monitor

dial = EmotionDial("Qwen/Qwen3-0.6B", layer=16)
dial.extract()                       # 抽方向

with dial:                           # 自动 attach / detach
    dial.set(happy=1.0, sad=-0.5)    # 拧旋钮
    print(dial.respond("你好呀"))

rows = localize(dial)                # 逐情绪定位层
res  = live_monitor(dial, layer=16)  # 逐 token 波动监测
```

## 原理（三张图说清）

1. **抽方向**：`系统提示=我此刻感到 E` 与 `中性系统提示` 下，目标层末位激活作差 → 向量 `v_E`。
2. **注入**：目标层 forward hook：`h ← h + Σ alpha_i · v_i`，alpha 可实时改、可逐 token 改。
3. **只读定位**：自然场景（非同一模板）→ 每句减中性质心 → 逐层算「自洽度 × 专属度」。
4. **只读监测**：情绪方向做 SVD → 取 3 个正交主成分当情绪子空间 → 逐 token 算投影与活跃度。

详细推导与坑位见 [`docs/METHOD.md`](docs/METHOD.md)。

## 它读到了什么？——一组参考发现

在一台本地机器上对 **Qwen3.5-4B（32 层，4bit）** 的观察（详见 [`docs/FINDINGS.md`](docs/FINDINGS.md)）：

- 情绪信息集中在**中层偏后（约 50%~66% 深度）**，峰值 L20–L21；
- 注入有台阶：**<0.45 隐形，0.45~0.6 甜蜜，>0.7 崩坏/复读**；
- 强情绪注入会形成**输出吸引子**（输入变、输出不动）；
- 「无力感 ≈ 恐惧 + 悲伤」；
- 存在一条**独立的「性欲」轴**，但表现得像一个被安全机制守护的「内容类别」，而非情绪旋钮。

## 目录结构

```
src/emotion_dial/
  engine.py     # 核心：加载 / 抽方向 / 注入 / 生成
  emotions.py   # 情绪提示词 + 自然场景语料
  probe.py      # 只读：逐层定位 + 逐 token 监测
  viz.py        # 热力图 / 曲线图
  cli.py        # 命令行入口
examples/       # 三个可跑示例
docs/           # METHOD.md（方法） + FINDINGS.md（实测发现）
```

## 说明与边界

- 这是**可解释性 / 行为研究**工具，**不是安全护栏**，也**不是越狱工具**。请遵守所用模型的服务条款与许可。
- 「情绪」是方便的比喻：本项目读写的是一阶线性方向，**不等于**模型真的有情绪体验。
- 方向是线性近似，4bit 量化会压低部分度量；定性结论通常比定量数字稳。
- Windows 老控制台（conhost）看中文可能乱码，属**显示问题**（非数据问题）；建议用 Windows Terminal 或 `chcp 65001`。

## 许可

[MIT](LICENSE)。

## 致谢 / 相关方向

思路受几条公开研究线启发：activation addition（Turner 等）、representation engineering（Zou 等）、
以及各家「steering vector」实践。本项目只是把它们整理成一个可上手的小工具，并附上一组本地实测。
