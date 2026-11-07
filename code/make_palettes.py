"""生成色板参考图 figures/palettes/*/palettes_<low|mid|high>.*：按饱和度分组，每种风格的四个基础色与两条热力图色带。

    python make_palettes.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from bilozo_style import FIG_DIR, GROUPS, STYLES, save, use_style

ZH = dict(fontfamily="Microsoft YaHei")
ROLES = [("main", "主数据"), ("accent", "强调"), ("gray", "背景个体"), ("ref", "参考线")]


def sheet(group):
    names = [k for k, v in STYLES.items() if v.get("group", "low") == group]
    use_style("sans")
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(6.875, 1.05 * n))
    grad = np.linspace(0, 1, 256)[None, :]
    for ax, name in zip(np.atleast_1d(axes), names):
        S = use_style(name)
        ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0, 0.97, f"{name}", fontsize=9, fontweight="bold", va="top", fontfamily="Arial")
        ax.text(1.0, 0.95, S.desc, fontsize=7, va="top", **ZH)
        for i, (attr, zh) in enumerate(ROLES):
            c = getattr(S, attr)
            x = i * 1.3
            ax.add_patch(Rectangle((x, 0.3), 1.15, 0.36, color=c, ec="#999999", lw=0.3))
            ax.text(x + 0.575, 0.48, zh, ha="center", va="center", fontsize=6.2,
                    color="white" if attr in ("main", "ref") else "#222222", **ZH)
            ax.text(x + 0.575, 0.24, f"S.{attr}\n{c}", ha="center", va="top", fontsize=5.6,
                    fontfamily="Arial", linespacing=1.1)
        for j, (cm, lab) in enumerate(((S.seq, "S.seq  连续量"), (S.seq_alert, "S.seq_alert  残差"))):
            y0 = 0.5 - 0.3 * j
            # 用 pcolormesh 而不是 imshow，色带保持矢量
            ax.pcolormesh(np.linspace(7.1, 9.9, 257), [y0, y0 + 0.16], grad, cmap=cm,
                          shading="flat", rasterized=False)
            ax.text(7.03, y0 + 0.08, lab, ha="right", va="center", fontsize=6, **ZH)
        ax.text(7.1, 0.12, "低", fontsize=5.6, va="top", **ZH)
        ax.text(9.9, 0.12, "高", fontsize=5.6, va="top", ha="right", **ZH)
    fig.subplots_adjust(hspace=0.35)
    fig.suptitle(f"{GROUPS[group]}（{group}）", x=0.0, ha="left", fontsize=9, fontweight="bold",
                 y=0.96, **ZH)
    save(fig, f"palettes_{group}", FIG_DIR / "palettes")


def main():
    for g in GROUPS:
        sheet(g)
    print("done")


if __name__ == "__main__":
    main()
