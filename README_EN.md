# emotion-dial

> Find, read, and **steer** emotion directions inside large language models — no training, no reloading the model. Turn one number to change the vibe.

`emotion-dial` is a small interpretability / activation-steering toolkit. It locates "emotion directions"
in the residual stream: it can **read** what's happening inside a model (per-layer localization, per-token
drift), and it can **inject** a vector to push the model toward an emotion. Built for Qwen3 / Qwen3.5 and
generic HuggingFace causal LMs.

[中文 README →](README.md)

---

## Highlights

- **Zero training**: compare system prompts → take the activation difference → hook and add.
- **An emotion workbench**:
  - `extract` — pull raw direction vectors for 8 emotions (savable/reusable);
  - `repl` — interactively dial `alpha`, like knobs on a mixer;
  - `localize` — find *which layer* each emotion is "brightest" at;
  - `monitor` — watch per-token internal emotion drift during generation.
- **Visualization included**: localization heatmaps and per-token triple plots, saved as PNG.
- **Model-agnostic**: auto-detects decoder layers; Qwen3 / Qwen3.5 / multimodal wrappers; 4-bit loading.

## Install

```bash
git clone https://github.com/<you>/emotion-dial.git
cd emotion-dial
pip install -e .            # or: pip install -e ".[viz]"  (adds matplotlib)
```

Deps: `torch`, `transformers`, `numpy` (`matplotlib` for plots, `bitsandbytes` for 4-bit).

## Quickstart (CLI)

```bash
emotion-dial extract  --model Qwen/Qwen3-0.6B --layer 16 --save vecs.pt
emotion-dial demo     --model Qwen/Qwen3-0.6B --layer 16 --load-vectors vecs.pt --prompt "Say hi!"
emotion-dial repl     --model Qwen/Qwen3-0.6B --layer 16
emotion-dial localize --model Qwen/Qwen3.5-4B --load-4bit --out localize.png --json localize.json
emotion-dial monitor  --model Qwen/Qwen3.5-4B --layer 21 --load-4bit --out live.png
```

## Quickstart (Python)

```python
from emotion_dial import EmotionDial
from emotion_dial.probe import localize, live_monitor

dial = EmotionDial("Qwen/Qwen3-0.6B", layer=16)
dial.extract()

with dial:                          # auto attach / detach
    dial.set(happy=1.0, sad=-0.5)   # turn the knobs
    print(dial.respond("Hello!"))

rows = localize(dial)               # per-emotion layer localization
res  = live_monitor(dial, layer=16) # per-token drift
```

## How it works

1. **Extract** — difference between activations under `system: you feel E` and a neutral system prompt
   at the target layer → direction `v_E`.
2. **Steer** — forward hook at the target layer: `h ← h + Σ alpha_i · v_i`; alphas are live-adjustable.
3. **Read (localize)** — natural scenes (not one shared template) → subtract neutral centroid →
   per-layer `consistency × uniqueness`.
4. **Read (monitor)** — SVD over emotion directions → 3 orthogonal PCs = emotion subspace → per-token
   projection and "emotional activity".

See [`docs/METHOD.md`](docs/METHOD.md).

## A reference observation

Local runs on **Qwen3.5-4B (32 layers, 4-bit)** (see [`docs/FINDINGS.md`](docs/FINDINGS.md)):

- emotion info concentrates in **mid-to-late layers (~50–66% depth)**, peaking at L20–L21;
- steering has steps: **<0.45 invisible, 0.45–0.6 sweet spot, >0.7 collapse/looping**;
- strong emotion injects form an **output attractor** (input changes, output doesn't);
- "helplessness ≈ fear + sadness";
- there is a **distinct "lust" axis**, but it behaves like a safety-guarded *content category*, not a knob.

## Layout

```
src/emotion_dial/  engine.py  emotions.py  probe.py  viz.py  cli.py
examples/          three runnable scripts
docs/              METHOD.md  FINDINGS.md
```

## Notes & scope

- This is a research/interpretability tool, **not a safety guardrail and not a jailbreak tool**.
  Respect the terms and licenses of the models you use.
- "Emotion" is a useful metaphor: the toolkit reads/writes first-order linear directions, which is
  **not** evidence of felt experience.
- Directions are linear approximations; 4-bit quantization deflates some metrics. Qualitative findings
  tend to be more stable than exact numbers.

## License

[MIT](LICENSE).

## Related work

Inspired by activation addition (Turner et al.), representation engineering (Zou et al.), and the broader
"steering vector" practice. This repo just packages a hands-on tool plus a set of local measurements.
