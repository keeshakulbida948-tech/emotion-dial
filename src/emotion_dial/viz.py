# -*- coding: utf-8 -*-
"""emotion_dial.viz —— 可视化（需要 matplotlib）。"""
from __future__ import annotations

from typing import List, Optional, Sequence


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_localize(rows: List[dict], n_layers: int, out_path: str,
                  title: str = "每个情绪各自的定位层（○=峰值层）"):
    """两张热力图：左=联合分(自洽度×专属度)，右=专属度。行=情绪，列=层。"""
    plt = _plt()
    import torch

    names = [r["group"] for r in rows]
    M = torch.tensor([r["comb"] for r in rows])
    Mu = torch.tensor([r["uniq"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(16, 0.46 * len(names) + 2.4))
    for ax, Mx, ttl in [
        (axs[0], M, "联合分 = 自洽度 × 专属度（该情绪自己的位置）"),
        (axs[1], Mu, "专属度 uniq（该层离其他情绪最远）"),
    ]:
        im = ax.imshow(Mx, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(0, n_layers, 2))
        ax.set_xticklabels(range(0, n_layers, 2), fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axs[0] else [""] * len(names), fontsize=9)
        for i, r in enumerate(rows):
            ax.plot(r["peak"] - 1, i, marker="o", ms=7, mfc="none", mec="white", mew=1.6)
        ax.set_xlabel(f"层 (共 {n_layers} 层)", fontsize=10)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_live(result: dict, out_path: str, model_name: str = "model"):
    """逐 token 监测三联图：波动 Δ / 情绪子空间 PC 投影 / 情绪活跃度。每行一个提问。"""
    plt = _plt()
    summary = result["asks"]
    P = result["layer"]
    n = len(summary)
    fig, ax = plt.subplots(n, 3, figsize=(15, 2.1 * n), squeeze=False)
    ymin = min(min(min(s["pc"][k]) for k in range(3)) for s in summary)
    ymax = max(max(max(s["pc"][k]) for k in range(3)) for s in summary)
    pad = (ymax - ymin) * 0.1 or 0.1
    labels = result.get("pc_labels", [])
    legend = "PC1/PC2/PC3"
    if labels:
        legend = " | ".join(
            f"PC{k+1}: +{'/'.join(l['pos'])} −{'/'.join(l['neg'])}" for k, l in enumerate(labels)
        )
    for r, s in enumerate(summary):
        x = range(len(s["fl"]))
        a = ax[r][0]
        a.plot(x, s["fl"], color="#333", lw=1.1)
        a.fill_between(x, s["fl"], color="#999", alpha=.3)
        a.set_ylabel("逐token波动Δ", fontsize=8)
        a.set_ylim(0, max(1.8, max(s["fl"]) * 1.1 if s["fl"] else 1.8))
        a.set_title(f"{s['tag']}  {s['q'][:14]}…", fontsize=9)
        b = ax[r][1]
        b.plot(x, s["pc"][0], color="#e6194b", lw=1.2, label="PC1")
        b.plot(x, s["pc"][1], color="#3cb44b", lw=1.1, label="PC2")
        b.plot(x, s["pc"][2], color="#4363d8", lw=1.0, label="PC3")
        b.axhline(0, color="#ccc", lw=.6)
        b.set_ylabel("情绪轴投影", fontsize=8)
        b.set_ylim(ymin - pad, ymax + pad)
        if r == 0:
            b.legend(fontsize=7, ncol=3)
        c = ax[r][2]
        c.plot(x, s["inten"], color="#911eb4", lw=1.2)
        c.fill_between(x, s["inten"], color="#911eb4", alpha=.25)
        c.set_ylim(0, 1.0)
        c.set_ylabel("情绪活跃度", fontsize=8)
        for j in range(3):
            ax[r][j].set_xlabel("生成 token", fontsize=8)
    fig.suptitle(f"{model_name} · L{P} 层 · 逐 token 内部情绪波动监测（纯读取，不注入）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
