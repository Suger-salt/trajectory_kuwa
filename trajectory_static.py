# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



### 軌跡をすべてプロットした形で見たい。
# ====== 設定（ここだけ触ればOK） ======
FILE = "2025_1025_dobusarai.csv"
POINT = "kuwa - 1"  # 例: "kuwa - 2"
WINDOWS = [(0, 10), (10,15),(15, 20),(20,28),(28,35)]  # 比較したい時間スパンのリスト（(start, end)）
# 片方を None にすると端まで：例) [(None, 10), (10, None)]
# =====================================

# --- 読み込み & 数値化 ---
df = pd.read_csv(FILE)
t = pd.to_numeric(df["time"], errors="coerce")
x = pd.to_numeric(df[f"{POINT}_x"], errors="coerce")
z = pd.to_numeric(df[f"{POINT}_z"], errors="coerce")

finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(z)
t, x, z = t[finite].to_numpy(), x[finite].to_numpy(), z[finite].to_numpy()

# --- 図の用意 ---
n = len(WINDOWS)
fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
axes = axes[0]

def _set_equal_margin(ax, a, b, m=0.05):
    if len(a) < 2 or len(b) < 2:
        ax.autoscale(); return
    amin, amax = np.nanmin(a), np.nanmax(a)
    bmin, bmax = np.nanmin(b), np.nanmax(b)
    ar = (amax - amin) if amax > amin else 1.0
    br = (bmax - bmin) if bmax > bmin else 1.0
    ax.set_xlim(amin - m*ar, amax + m*ar)
    ax.set_ylim(bmin - m*br, bmax + m*br)
    ax.set_aspect("equal", adjustable="box")


# ラベル用にハンドルを保存するリスト
handles, labels = [], []

for ax, (start, end) in zip(axes, WINDOWS):
    # 時間範囲抽出
    t0 = start if start is not None else np.nanmin(t)
    t1 = end   if end   is not None else np.nanmax(t)
    if t0 > t1:
        t0, t1 = t1, t0
    sel = (t >= t0) & (t <= t1)
    tx, xx, zz = t[sel], x[sel], z[sel]

    if len(tx) < 2:
        ax.set_title(f"{POINT} | XZ  ({t0}-{t1}s)\nデータ不足")
        ax.grid(True)
        continue

    # プロット
    line, = ax.plot(xx, zz, lw=2, label="trajectory")
    start_pt, = ax.plot(xx[0], zz[0], "o", ms=6, label="start")
    end_pt, = ax.plot(xx[-1], zz[-1], "s", ms=6, label="end")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_title(f"{POINT} | XZ  ({tx[0]:.2f}-{tx[-1]:.2f}s)")
    ax.grid(True)
    _set_equal_margin(ax, xx, zz)

    # 最初のサブプロットだけから凡例ハンドルを取得
    if not handles:
        handles = [line, start_pt, end_pt]
        labels = [h.get_label() for h in handles]

# --- 共通凡例を図全体に表示 ---
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # 凡例のために上に少し余白
plt.show()
