"""BiLoZO 论文绘图风格模板。

用法：
    from bilozo_style import use_style, panel_label, save
    S = use_style("sans")                  # 七种低饱和风格见 STYLES
    fig, ax = plt.subplots(figsize=(S.col1, 2.0))
    ax.plot(x, y, color=S.main)
    panel_label(ax, "a", S)
    save(fig, "my_figure")                 # -> figures/pdf, figures/svg（矢量）, figures/png（预览）

尺寸按 CVPR 双栏排版：单栏宽 3.25 in，通栏宽 6.875 in。
所有字号都是最终印刷尺寸下的 pt，插入 LaTeX 时用 width=\\linewidth，不要再缩放。
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap

COL1 = 3.25    # CVPR 单栏宽（英寸）
COL2 = 6.875   # CVPR 通栏宽（英寸）
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


@dataclass
class Style:
    name: str
    desc: str          # 一句话说明
    font: str          # 全图字体
    main: str          # 主数据：主曲线、主散点
    accent: str        # 强调：对照方法、阈值线、低于阈值的点
    gray: str          # 背景：逐层 / 逐矩阵的个体曲线
    ref: str           # 参考：随机基线、τ 虚线、中位数横线
    seq: Colormap      # 连续量热力图（浅=低，深=高），如 E(32)、ρ_t
    seq_alert: Colormap  # “缺失 / 残差”热力图（浅=小，深=大），如 1−E(32)
    seq_stops: object  # seq 的色标（十六进制列表）或 matplotlib 配色名
    alert_stops: object
    grid: bool         # 是否画浅灰网格
    group: str = "low"  # 饱和度分组：low / mid / high
    col1: float = COL1
    col2: float = COL2


def _cmap(stops, name):
    if isinstance(stops, str):
        return plt.get_cmap(stops)
    return LinearSegmentedColormap.from_list(name, stops)


# 三组配色，按饱和度分：low（低）/ mid（中）/ high（高）。都不用 viridis / jet 这类彩虹色带，
# 热力图色带一律是“近白 → 单一色相加深”。
GROUPS = {"low": "低饱和", "mid": "中饱和", "high": "高饱和"}

STYLES: dict[str, dict] = {
    "morandi": dict(
        desc="莫兰迪风：灰蓝配灰粉，饱和度低、偏暖灰，整体最安静",
        font="Arial", main="#5F7889", accent="#B07A6C", gray="#D9D4CC", ref="#5A5550",
        seq_stops=["#F6F3EE", "#DCD6CA", "#B3B8AE", "#879A9C", "#5B7180"],
        alert_stops=["#FBF8F4", "#EAD9CF", "#CFA99B", "#AA7869", "#7A5045"], grid=False),
    "tol": dict(
        desc="Paul Tol muted：学术常用的低饱和色盲友好配色，青绿配玫瑰",
        font="Arial", main="#44AA99", accent="#CC6677", gray="#DDDDDD", ref="#555555",
        seq_stops=["#F7F5EF", "#D3E4DF", "#9CC7BD", "#5FA597", "#33766B"],
        alert_stops=["#FBF6F6", "#EFD0D4", "#D99AA5", "#CC6677", "#882255"], grid=False),
    "nord": dict(
        desc="Nord 极地风：冷灰蓝为主，极光红做强调，干净偏冷",
        font="Arial", main="#5E81AC", accent="#BF616A", gray="#D8DEE9", ref="#4C566A",
        seq_stops=["#ECEFF4", "#CBD6E2", "#9DBBCB", "#81A1C1", "#4F6E96"],
        alert_stops=["#F6F2EA", "#EBCB8B", "#D08770", "#BF616A", "#8A4A52"], grid=False),
    "sage": dict(
        desc="鼠尾草风：灰绿配驼色，自然沉静",
        font="Arial", main="#5F7A6A", accent="#B5835A", gray="#D5D8D0", ref="#4F4F4F",
        seq_stops=["#F5F4EE", "#DDE2D3", "#AFC0A6", "#7D9A84", "#4F6B5C"],
        alert_stops=["#FBF6EF", "#EEDCC4", "#D4B08A", "#B5835A", "#7D5636"], grid=False),
    "sans": dict(
        desc="雾蓝简洁风：Arial，降低饱和度的蓝配陶土色，最接近原图",
        font="Arial", main="#3E6A8F", accent="#C27A4A", gray="#CDD2D8", ref="#3A3A3A",
        seq_stops=["#F4F6F8", "#CBD7E1", "#93AEC4", "#5F86A6", "#34587A"],
        alert_stops=["#FBF7F3", "#F0DCCB", "#DDAE8A", "#C27A4A", "#86502C"], grid=False),
    "serif": dict(
        desc="衬线论文风：Times New Roman，与 CVPR 正文字体一致，深藏青配砖红",
        font="Times New Roman", main="#2F4568", accent="#A0524D", gray="#CCCCCC", ref="#222222",
        seq_stops=["#F7F7F2", "#D2DACB", "#98AE9C", "#5E7F79", "#35525A"],
        alert_stops=["#FBF6F4", "#EED6D1", "#CF9A92", "#A0524D", "#6A2E2B"], grid=False),
    "mono": dict(
        desc="黑白打印风：灰阶为主，只留一个灰红强调，打印成黑白也能读",
        font="Arial", main="#2B2B2B", accent="#A9544A", gray="#CFCFCF", ref="#666666",
        seq_stops=["#F5F5F5", "#CDCDCD", "#9A9A9A", "#626262", "#333333"],
        alert_stops=["#FAF7F6", "#E9D0CC", "#CB938B", "#A9544A", "#6B302A"], grid=False),

    # ---------------------------------------------------------------- 中饱和
    "ocean": dict(
        group="mid", desc="海蓝风：标准学术蓝配橙，比 sans 更有精神但不刺眼",
        font="Arial", main="#2E6F9E", accent="#D17A45", gray="#CBD3DB", ref="#333333",
        seq_stops=["#F3F6F9", "#BFD3E6", "#7FA8CC", "#3F7AAE", "#1F4E7A"],
        alert_stops=["#FDF6EE", "#F6D2B0", "#E7A06C", "#C8683A", "#8A3F1E"], grid=False),
    "teal": dict(
        group="mid", desc="青绿风：青绿配珊瑚红，清爽，适合幻灯片和论文共用",
        font="Arial", main="#2A8C82", accent="#D0675E", gray="#D2D8D6", ref="#3A3A3A",
        seq_stops=["#F4F8F6", "#C6E2DC", "#84C2B6", "#3E9A8C", "#1E6158"],
        alert_stops=["#FDF5F3", "#F4CDC6", "#E59A8F", "#CC6156", "#8E3A33"], grid=False),
    "plum": dict(
        group="mid", desc="紫金风：灰紫配琥珀，辨识度高，和常见蓝橙图拉开差别",
        font="Arial", main="#6A5A9E", accent="#D49A3A", gray="#D6D4DE", ref="#3A3A3A",
        seq_stops=["#F6F4FA", "#D6D0E8", "#A99CD0", "#7A69B0", "#4D3F80"],
        alert_stops=["#FEF8EC", "#F7E1AE", "#EBBE62", "#D49A3A", "#95651C"], grid=False),
    "okabe": dict(
        group="mid", desc="Okabe–Ito 色盲友好：Nature 等期刊推荐的经典配色",
        font="Arial", main="#0072B2", accent="#D55E00", gray="#BFBFBF", ref="#222222",
        seq_stops=["#F5F9FC", "#B5D3E8", "#6AA6D1", "#0072B2", "#004A75"],
        alert_stops=["#FFF8EF", "#F5D7A6", "#E69F00", "#D55E00", "#8A3D00"], grid=False),

    # ---------------------------------------------------------------- 高饱和
    "vivid": dict(
        group="high", desc="鲜明蓝红：纯度高、对比强，投影和海报上最醒目",
        font="Arial", main="#0063B2", accent="#E4572E", gray="#C4C9D0", ref="#222222",
        seq_stops=["#F2F7FC", "#A9CDF0", "#4E97DD", "#1565C0", "#0B3D7A"],
        alert_stops=["#FFF5EB", "#FDC58A", "#F7893B", "#E4572E", "#A52A0F"], grid=False),
    "tableau": dict(
        group="high", desc="Tableau 10：matplotlib 默认色（tab:blue / tab:orange），读者最熟悉",
        font="Arial", main="#1F77B4", accent="#FF7F0E", gray="#C7C7C7", ref="#333333",
        seq_stops=["#F1F7FC", "#AED1EE", "#5AA3DA", "#1F77B4", "#0E4772"],
        alert_stops=["#FFF6EC", "#FFD0A1", "#FFA04D", "#FF7F0E", "#B3540A"], grid=False),
    "npg": dict(
        group="high", desc="NPG 风：Nature 出版集团配色（ggsci），翠绿配朱红",
        font="Arial", main="#00A087", accent="#E64B35", gray="#C9CFCD", ref="#333333",
        seq_stops=["#EFF8F6", "#A8DCD1", "#4DBBAA", "#00A087", "#00634F"],
        alert_stops=["#FEF3F0", "#F8BFB2", "#F08A73", "#E64B35", "#9C2A1A"], grid=False),
}


def use_style(name="sans"):
    """设置全局 rcParams，并返回该风格的配色对象 S。"""
    cfg = dict(STYLES[name])
    cfg["seq"] = _cmap(cfg["seq_stops"], f"{name}_seq")
    cfg["seq_alert"] = _cmap(cfg["alert_stops"], f"{name}_alert")
    S = Style(name=name, **cfg)
    mpl.rcParams.update({
        "font.family": S.font,
        "mathtext.fontset": "stix" if S.font == "Times New Roman" else "dejavusans",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": S.grid,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        # 矢量导出：字体嵌入为可编辑文字，SVG 保留文字而不是转成路径
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "image.composite_image": False,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })
    return S


def panel_label(ax, letter, S, x=-0.16, y=1.04):
    """在子图左上角写小写粗体面板字母（a, b, c ...）。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, name, fig_dir=FIG_DIR):
    """导出矢量图 PDF（插 LaTeX 用）、SVG（可在 Illustrator/Inkscape 里改）和 PNG 预览。"""
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        out = Path(fig_dir) / ext / f"{name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
