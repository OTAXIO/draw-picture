"""读取 绘图/data 下的 spectra.csv 与 capture.csv，整理成绘图用的数组。

数据复制自 BiloZO/数据（Qwen3-0.6B / SST-2，左秩 32，右秩 1），实验说明见 data/figure_notes.txt。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MODULE_LABELS = ["q", "k", "v", "o", "gate", "up", "down"]
# Qwen3-0.6B 各投影矩阵的输出维度（行数 m），用于随机子空间的期望 k/m
ROWS = {"q_proj": 2048, "k_proj": 1024, "v_proj": 1024, "o_proj": 1024,
        "gate_proj": 3072, "up_proj": 3072, "down_proj": 1024}
N_LAYERS = 28


def _module_of(matrix):
    return matrix.split(".")[-2]


def load_spectra(path=DATA_DIR / "spectra.csv"):
    """返回 {matrix: dict(layer, module, sv, cum)}，sv、cum 按 k=1,2,... 排序。"""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["matrix"]].append((int(r["k"]), float(r["singular_value"]),
                                      float(r["cumulative_energy"]), int(r["layer"]), r["module"]))
    out = {}
    for m, lst in rows.items():
        lst.sort()
        out[m] = dict(layer=lst[0][3], module=lst[0][4],
                      sv=np.array([x[1] for x in lst]), cum=np.array([x[2] for x in lst]))
    return out


def energy_grid(spectra, rank=32):
    """E(rank) 矩阵，形状 (28 层, 7 模块)。"""
    E = np.full((N_LAYERS, len(MODULES)), np.nan)
    for d in spectra.values():
        E[d["layer"], MODULES.index(d["module"])] = d["cum"][min(rank, len(d["cum"])) - 1]
    return E


def load_capture(path=DATA_DIR / "capture.csv", rank=32):
    """返回 steps、全局能量加权 rho、矩阵中位数 rho、逐层 rho（层 × 步）和随机基期望。"""
    num, den = defaultdict(dict), defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = int(r["step"])
            num[s][r["matrix"]] = float(r["numerator"])
            den[s][r["matrix"]] = float(r["denominator"])
    steps = np.array(sorted(num))
    glob, med, rand = [], [], []
    layer = np.zeros((N_LAYERS, len(steps)))
    for j, s in enumerate(steps):
        mats = list(num[s])
        n = np.array([num[s][m] for m in mats])
        d = np.array([den[s][m] for m in mats])
        glob.append(n.sum() / d.sum())
        med.append(np.median(n / d))
        kk = np.array([rank / ROWS[_module_of(m)] for m in mats])
        rand.append((kk * d).sum() / d.sum())
        lay = np.array([int(m.split(".")[2]) for m in mats])
        for l in range(N_LAYERS):
            sel = lay == l
            layer[l, j] = n[sel].sum() / d[sel].sum()
    return dict(steps=steps, glob=np.array(glob), med=np.array(med),
                rand=np.array(rand), layer=layer)
