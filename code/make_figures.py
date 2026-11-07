"""用真实数据重绘 Figure 1（跨 batch 低秩）与 Figure 2（固定 B0 的捕获率），每个子图单独输出。

    python make_figures.py                                   # 默认：vivid 配色，子图 c 用 diverge
    python make_figures.py --style morandi --panel-c outline # 换配色 / 换子图 c 画法
    python make_figures.py --labels                          # 小图左上角加 a / b / c
    python make_figures.py --options                         # 额外生成子图 c 九种画法的对比页

输出：
    figures/fig1/{pdf,svg,png}/fig1a, fig1b, fig1c
    figures/fig2/{pdf,svg,png}/fig2a, fig2b
    figures/options/...（仅 --options）
pdf、svg 为矢量图，png 为 300 dpi 预览。
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec

from bilozo_data import MODULE_LABELS, N_LAYERS, energy_grid, load_capture, load_spectra
from bilozo_style import FIG_DIR, GROUPS, STYLES, panel_label, save, use_style

RANK, TAU = 32, 0.9
ZH = dict(fontfamily="Microsoft YaHei")   # 仅用于对比页上的中文说明


# ---------------------------------------------------------------- 公共小工具
def _colorbar(fig, ax, mappable, label, ticks=None, horizontal=False):
    """色条放在子图外侧的内嵌坐标轴里，不挤占子图本身的大小。"""
    if horizontal:
        cax = ax.inset_axes([0.0, 1.06, 1.0, 0.045])
        cb = fig.colorbar(mappable, cax=cax, ticks=ticks, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        # 标签写在色条左侧，避免与上方刻度重叠
        cax.text(-0.02, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=6.5)
    else:
        cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0])
        cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
        cb.set_label(label, labelpad=2)
    cb.solids.set_rasterized(False)   # matplotlib 默认把色条栅格化，这里保持矢量
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4)
    cax.grid(False)
    return cb


def _truncate(cmap, hi=0.85):
    """去掉配色最深的一段，避免大面积高值时整块发黑。"""
    return ListedColormap(cmap(np.linspace(0, hi, 256)))


def _gaps(ax, n, axis="x"):
    """只在模块之间画白色细缝，避免 28 层变成一排横条纹。"""
    for v in range(1, n):
        (ax.axvline if axis == "x" else ax.axhline)(v, color="white", lw=1.2)


def _heat_axes(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)


def _layer_ticks(ax, axis="y"):
    ticks = np.arange(0, N_LAYERS, 5)
    if axis == "y":
        ax.set_yticks(ticks + 0.5, [str(t) for t in ticks]); ax.set_ylabel("Decoder layer")
    else:
        ax.set_xticks(ticks + 0.5, [str(t) for t in ticks]); ax.set_xlabel("Decoder layer")


def _elim(E):
    return np.floor(np.nanmin(E) * 20) / 20


def _below_note(ax, S, x=1.065, y=-0.07):
    ax.scatter([x], [y], s=5, color=S.accent, transform=ax.transAxes, clip_on=False)
    ax.text(x + 0.035, y, f"< {TAU}", transform=ax.transAxes, va="center", fontsize=6)


# ---------------------------------------------------------------- 子图 c 的九种画法（都是面积型）
def c_heat(fig, ax, E, S):
    """改进热力图：色阶从数据最小值起，模块间竖缝，低于 τ 的格子打点。"""
    im = ax.pcolormesh(np.arange(8), np.arange(N_LAYERS + 1), E, cmap=_truncate(S.seq),
                       norm=Normalize(_elim(E), 1.0), edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "x")
    ys, xs = np.where(E < TAU)
    ax.scatter(xs + 0.5, ys + 0.5, s=5, color=S.accent, edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    _colorbar(fig, ax, im, f"E({RANK})")
    _below_note(ax, S)


def c_heat_t(fig, ax, E, S):
    """横向热力图：层放在横轴，7 个模块成 7 行，色条放在上方。"""
    im = ax.pcolormesh(np.arange(N_LAYERS + 1), np.arange(8), E.T[::-1], cmap=_truncate(S.seq),
                       norm=Normalize(_elim(E), 1.0), edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "y")
    xs, ys = np.where(E.T[::-1] < TAU)
    ax.scatter(ys + 0.5, xs + 0.5, s=4, color=S.accent, edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_yticks(np.arange(7) + 0.5, MODULE_LABELS[::-1])
    ax.set_xlim(0, N_LAYERS); ax.set_ylim(0, 7)
    _layer_ticks(ax, "x"); _heat_axes(ax)
    ax.set_ylabel("Projection module")
    _colorbar(fig, ax, im, f"E({RANK})", horizontal=True)


def c_residual(fig, ax, E, S):
    """残差热力图：画没被捕获的能量 1−E(32)，对数色阶把差异放大。"""
    R = np.clip(1 - E, 1e-4, None)
    norm = LogNorm(vmin=max(1e-3, np.nanmin(R)), vmax=np.nanmax(R))
    im = ax.pcolormesh(np.arange(8), np.arange(N_LAYERS + 1), R, cmap=S.seq_alert, norm=norm,
                       edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "x")
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    _colorbar(fig, ax, im, f"1 − E({RANK})")


def _outline_low(ax, E, S, x_of=lambda j: j, lw=0.8):
    """低于 τ 的格子用强调色描边，比打点更利落。"""
    for i, j in zip(*np.where(E < TAU)):
        ax.add_patch(Rectangle((x_of(j), i), 1, 1, fill=False, ec=S.accent, lw=lw, zorder=3))


def _outline_note(ax, S, x=1.04, y=-0.075):
    ax.add_patch(Rectangle((x, y - 0.012), 0.045, 0.024, transform=ax.transAxes, fill=False,
                           ec=S.accent, lw=0.8, clip_on=False))
    ax.text(x + 0.065, y, f"< {TAU}", transform=ax.transAxes, va="center", fontsize=6)


def _text_color(rgba):
    """按格子亮度选黑字或白字（matplotlib 注释热力图示例的做法）。"""
    r, g, b = rgba[:3]
    return "#222222" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def c_outline(fig, ax, E, S):
    """描边热力图：与 heat 相同，但低于 τ 的格子用强调色方框圈出。"""
    im = ax.pcolormesh(np.arange(8), np.arange(N_LAYERS + 1), E, cmap=_truncate(S.seq),
                       norm=Normalize(_elim(E), 1.0), edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "x")
    _outline_low(ax, E, S)
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    _colorbar(fig, ax, im, f"E({RANK})")
    _outline_note(ax, S)


def c_grouped(fig, ax, E, S):
    """分组热力图：注意力（q k v o）与 MLP（gate up down）分成两块，下方用括号标注。"""
    xpos = np.array([0, 1, 2, 3, 4.45, 5.45, 6.45])
    norm, cm = Normalize(_elim(E), 1.0), _truncate(S.seq)
    for a, b in ((0, 4), (4, 7)):
        im = ax.pcolormesh(xpos[a] + np.arange(b - a + 1), np.arange(N_LAYERS + 1), E[:, a:b],
                           cmap=cm, norm=norm, edgecolors="face", linewidth=0.2)
    for v in (1, 2, 3, 5.45, 6.45):        # 只画模块间竖缝，不画横缝，避免横条纹
        ax.axvline(v, color="white", lw=1.2)
    _outline_low(ax, E, S, x_of=lambda j: xpos[j], lw=0.7)
    ax.set_xticks(xpos + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7.45); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    for (a, b), lab in (((0, 4), "Attention"), ((4.45, 7.45), "MLP")):
        ax.plot([a + 0.1, a + 0.1, b - 0.1, b - 0.1], [-3.9, -4.4, -4.4, -3.9], color=S.ref,
                lw=0.5, clip_on=False)
        ax.text((a + b) / 2, -4.8, lab, ha="center", va="top", fontsize=6.5, color=S.ref)
    _colorbar(fig, ax, im, f"E({RANK})")
    _outline_note(ax, S, y=-0.2)


def c_rounded(fig, ax, E, S):
    """圆角格子：每格一个圆角小方块，格间留白，观感接近日历热力图。"""
    norm, cm = Normalize(_elim(E), 1.0), _truncate(S.seq)
    pos = ax.get_position()
    w, h = fig.get_size_inches()
    aspect = (pos.width * w / 7) / (pos.height * h / N_LAYERS)   # 让圆角在纸面上是正圆
    for i in range(N_LAYERS):
        for j in range(7):
            low = E[i, j] < TAU
            ax.add_patch(FancyBboxPatch((j + 0.06, i + 0.1), 0.88, 0.8,
                                        boxstyle="round,pad=0,rounding_size=0.07",
                                        mutation_aspect=aspect, fc=cm(norm(E[i, j])),
                                        ec=S.accent if low else "none", lw=0.6))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    _colorbar(fig, ax, sm, f"E({RANK})")
    _outline_note(ax, S)


def c_block(fig, ax, E, S):
    """分段注释热力图：每 4 层合成一格，格内写该段最小 E(32)，数值直接可读。"""
    nb = N_LAYERS // 4
    B = E.reshape(nb, 4, 7).min(axis=1)
    norm, cm = Normalize(_elim(E), 1.0), _truncate(S.seq, 0.95)
    im = ax.pcolormesh(np.arange(8), np.arange(nb + 1), B, cmap=cm, norm=norm,
                       edgecolors="white", linewidth=1.0)
    for i in range(nb):
        for j in range(7):
            v = B[i, j]
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                    color=_text_color(cm(norm(v))), fontweight="bold" if v < TAU else "normal")
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_yticks(np.arange(nb) + 0.5, [f"{4 * i}–{4 * i + 3}" for i in range(nb)])
    ax.set_xlim(0, 7); ax.set_ylim(0, nb)
    _heat_axes(ax)
    ax.set_xlabel("Projection module"); ax.set_ylabel("Decoder layers")
    _colorbar(fig, ax, im, f"min E({RANK}) per 4 layers")


def c_diverge(fig, ax, E, S):
    """以 τ 为中心的发散色阶：低于 τ 用暖色，高于 τ 用冷色，阈值附近接近白色。"""
    warm = S.seq_alert(np.linspace(0.85, 0.0, 128))
    cool = S.seq(np.linspace(0.0, 0.85, 128))
    cm = ListedColormap(np.vstack([warm, cool]))
    lo = _elim(E)
    norm = TwoSlopeNorm(vmin=lo, vcenter=TAU, vmax=1.0)
    im = ax.pcolormesh(np.arange(8), np.arange(N_LAYERS + 1), E, cmap=cm, norm=norm,
                       edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "x")
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    _colorbar(fig, ax, im, f"E({RANK})", ticks=[lo, round((lo + TAU) / 2, 2), TAU, 0.95, 1.0])


def c_marginal(fig, ax, E, S):
    """热力图 + 边缘柱：上方为各模块平均缺失能量，右侧为各层平均缺失能量。"""
    R = 1 - E
    im = ax.pcolormesh(np.arange(8), np.arange(N_LAYERS + 1), E, cmap=_truncate(S.seq),
                       norm=Normalize(_elim(E), 1.0), edgecolors="face", linewidth=0.2)
    _gaps(ax, 7, "x")
    _outline_low(ax, E, S, lw=0.6)
    ax.set_xticks(np.arange(7) + 0.5, MODULE_LABELS)
    ax.set_xlim(0, 7); ax.set_ylim(0, N_LAYERS)
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Projection module")
    top = ax.inset_axes([0, 1.03, 1, 0.16])
    top.bar(np.arange(7) + 0.5, R.mean(0), width=0.78, color=S.main, alpha=0.8, lw=0)
    top.set_xlim(0, 7)
    right = ax.inset_axes([1.03, 0, 0.22, 1])
    right.barh(np.arange(N_LAYERS) + 0.5, R.mean(1), height=0.8, color=S.main, alpha=0.8, lw=0)
    right.set_ylim(0, N_LAYERS)
    for m in (top, right):
        m.grid(False)
        for s in ("top", "right"):
            m.spines[s].set_visible(False)
        for s in m.spines.values():
            s.set_linewidth(0.4)
        m.tick_params(labelsize=5.2, length=1.5, width=0.4)
    top.tick_params(labelbottom=False, bottom=False)
    top.set_yticks([0, round(float(R.mean(0).max()), 2)])
    top.set_ylabel("1−E", fontsize=5.5, labelpad=1)
    right.tick_params(labelleft=False, left=False)
    right.set_xticks([round(float(R.mean(1).max()), 2)])
    right.set_xlabel("1−E", fontsize=5.5, labelpad=1)
    cax = ax.inset_axes([1.32, 0, 0.045, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"E({RANK})", labelpad=2)
    cb.solids.set_rasterized(False); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4); cax.grid(False)


PANEL_C = {"heat": c_heat, "outline": c_outline, "grouped": c_grouped, "rounded": c_rounded,
           "block": c_block, "diverge": c_diverge, "marginal": c_marginal,
           "heat_t": c_heat_t, "residual": c_residual}
PANEL_C_ZH = {"heat": "heat  改进热力图：色阶从最小值起，低于 τ 打点",
              "outline": "outline  描边热力图：低于 τ 的格子描边",
              "grouped": "grouped  分组热力图：注意力 / MLP 分块",
              "rounded": "rounded  圆角格子：格间留白，日历热力图风",
              "block": "block  分段注释：每 4 层一格，格内写最小值",
              "diverge": "diverge  发散色阶：以 τ = 0.9 为中心",
              "marginal": "marginal  边缘柱：上 / 右为平均缺失能量",
              "heat_t": "heat_t  横向热力图：层在横轴，色条在上方",
              "residual": "residual  残差热力图：1−E(32)，对数色阶"}


# ---------------------------------------------------------------- 单张小图
# 每个子图单独成一张图。绘图区（坐标轴本身）的尺寸固定，几张图在 LaTeX 里并排时大小一致。
FIG1_AX = (1.6, 1.5)    # Figure 1 每个子图的绘图区宽、高（英寸）
FIG2_AX = (2.6, 1.6)    # Figure 2 每个子图的绘图区宽、高（英寸）


def _single(ax_size):
    """建一张只有一个坐标轴的图，绘图区为 ax_size 英寸；外围留白由 save 时的 bbox_inches='tight' 裁掉。"""
    w, h = ax_size
    W, H = w + 1.6, h + 1.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.7 / W, 0.5 / H, w / W, h / H])
    return fig, ax


def _label(ax, S, letter, x):
    if letter:
        panel_label(ax, letter, S, x=x)


def fig1a(spectra, S, letter=None):
    """Figure 1a：各矩阵的归一化奇异值谱，粗线为中位数。"""
    fig, ax = _single(FIG1_AX)
    sv = np.array([d["sv"] / d["sv"][0] for d in spectra.values()])
    k = np.arange(1, sv.shape[1] + 1)
    for row in sv:
        ax.semilogy(k, row, color=S.gray, lw=0.3, alpha=0.6)
    ax.semilogy(k, np.median(sv, axis=0), color=S.main, lw=1.5)
    ax.set_xlabel("Singular-value index k"); ax.set_ylabel(r"$\sigma_k/\sigma_1$")
    ax.set_title("Cross-batch singular spectrum")
    ax.set_xlim(0, k[-1] + 1)
    _label(ax, S, letter, -0.3)
    return fig


def fig1b(spectra, S, letter=None):
    """Figure 1b：累计能量曲线，τ 虚线与 k=32 竖线。"""
    fig, ax = _single(FIG1_AX)
    cum = np.array([d["cum"] for d in spectra.values()])
    k = np.arange(1, cum.shape[1] + 1)
    for row in cum:
        ax.plot(k, row, color=S.gray, lw=0.3, alpha=0.6)
    ax.plot(k, np.median(cum, axis=0), color=S.main, lw=1.5, label="Median over matrices")
    ax.axhline(TAU, color=S.ref, lw=0.6, ls=":")
    ax.axvline(RANK, color=S.accent, lw=0.9, ls="--", label=f"Left rank {RANK}")
    ax.set_xlabel("Left rank k"); ax.set_ylabel("Cumulative energy")
    ax.set_title("Cumulative energy")
    ax.set_xlim(0, k[-1] + 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _label(ax, S, letter, -0.3)
    return fig


def fig1c(spectra, S, variant="diverge", letter=None):
    """Figure 1c：E(32) 热力图，画法见 PANEL_C。"""
    fig, ax = _single(FIG1_AX)
    PANEL_C[variant](fig, ax, energy_grid(spectra, RANK), S)
    ax.set_title(f"Energy captured at rank {RANK}",
                 pad=24 if variant == "marginal" else 14 if variant == "heat_t" else 4)
    _label(ax, S, letter, -0.3)
    return fig


def fig2a(cap, S, letter=None):
    """Figure 2a：捕获率随训练步数的变化。step=1 处不画标记点，但保留 1→1000 的连线。"""
    fig, ax = _single(FIG2_AX)
    steps = cap["steps"]
    mk = slice(1, None)          # 从第二个点（step=1000）开始画标记
    for row in cap["layer"]:
        ax.plot(steps, row, color=S.gray, lw=0.45)
    ax.plot(steps, cap["glob"], color=S.main, lw=1.5, marker="o", ms=2.5, markevery=mk,
            label="Energy-weighted (all Linear)")
    ax.plot(steps, cap["med"], color=S.accent, lw=1.1, ls=":", marker="s", ms=2.2, markevery=mk,
            label="Median matrix")
    ax.plot(steps, cap["rand"], color=S.ref, lw=0.8, ls="--", label=f"Random rank-{RANK} (expected)")
    ax.plot([], [], color=S.gray, lw=0.8, label="Individual layers")
    ax.set_xlabel("Optimizer step"); ax.set_ylabel(r"Gradient capture $\rho_t$")
    ax.set_title(r"Fixed initialization subspace $B_0$")
    ax.set_ylim(0, 1.02); ax.set_xlim(-400, steps[-1] + 400)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    _label(ax, S, letter, -0.14)
    return fig


def fig2b(cap, S, letter=None):
    """Figure 2b：逐层捕获率热力图（层 × 训练步）。"""
    fig, ax = _single(FIG2_AX)
    steps = cap["steps"]
    im = ax.pcolormesh(np.arange(len(steps) + 1), np.arange(N_LAYERS + 1), cap["layer"],
                       cmap=S.seq, norm=Normalize(0, 1), edgecolors="face", linewidth=0.2)
    idx = np.arange(0, len(steps), 5)
    ax.set_xticks(idx + 0.5, [f"{steps[i] // 1000}k" if steps[i] > 1 else "0" for i in idx])
    _layer_ticks(ax); _heat_axes(ax)
    ax.set_xlabel("Optimizer step")
    ax.set_title("Per-layer energy-weighted capture")
    _colorbar(fig, ax, im, r"$\rho_t$", ticks=[0, 0.25, 0.5, 0.75, 1])
    _label(ax, S, letter, -0.14)
    return fig


# ---------------------------------------------------------------- 子图 c 对比页
def options_sheet(spectra, S):
    E = energy_grid(spectra, RANK)
    fig = plt.figure(figsize=(S.col2, 13.2))
    gs = GridSpec(5, 2, figure=fig, wspace=0.8, hspace=0.95)
    for i, v in enumerate(PANEL_C):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        PANEL_C[v](fig, ax, E, S)
        ax.set_title(PANEL_C_ZH[v], loc="left", fontsize=8, pad=26 if v == "marginal" else 16 if v == "heat_t" else 8, **ZH)
    note = fig.add_subplot(gs[4, 1]); note.axis("off")
    note.text(0, 0.6, f"风格：{S.name}（{GROUPS[S.group]}）\n{STYLES[S.name]['desc']}\n\n命令：\npython make_figures.py "
              f"--style {S.name} --panel-c <画法>", fontsize=7.5, va="top", **ZH)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=list(STYLES), default="vivid")
    p.add_argument("--panel-c", choices=list(PANEL_C), default="diverge")
    p.add_argument("--labels", action="store_true", help="在每张小图左上角写 a / b / c")
    p.add_argument("--options", action="store_true", help="额外生成子图 c 九种画法的对比页")
    a = p.parse_args()
    spectra, cap = load_spectra(), load_capture(rank=RANK)
    S = use_style(a.style)
    lab = (lambda c: c) if a.labels else (lambda c: None)

    d1, d2 = FIG_DIR / "fig1", FIG_DIR / "fig2"
    save(fig1a(spectra, S, lab("a")), "fig1a", d1)
    save(fig1b(spectra, S, lab("b")), "fig1b", d1)
    save(fig1c(spectra, S, a.panel_c, lab("c")), "fig1c", d1)
    save(fig2a(cap, S, lab("a")), "fig2a", d2)
    save(fig2b(cap, S, lab("b")), "fig2b", d2)
    if a.options:
        save(options_sheet(spectra, S), f"fig1c_options_{a.style}", FIG_DIR / "options")
    print("done")


if __name__ == "__main__":
    main()
