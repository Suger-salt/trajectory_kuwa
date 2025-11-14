# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### ここで時間を軸にした、速度や加速度の変化量を見てみたい。
# ====== 設定（ここを変える） ======
FILE    = "2025_1025_dobusarai.csv"
POINT   = "kuwa - 1"         # 例: "kuwa - 2"
AXIS    = "z"                # ← 'x' / 'y' / 'z' から選ぶ
T_START = None               # 例: 0.0（Noneなら最小時刻）
T_END   = 40.0               # 例: 30.0（Noneなら最大時刻）
# ===============================

# --- 読み込み & 数値化 ---
df = pd.read_csv(FILE)
t = pd.to_numeric(df["time"], errors="coerce")
pos = pd.to_numeric(df[f"{POINT}_{AXIS}"], errors="coerce")  # 選んだ軸の位置

# 有限値のみ抽出
finite = np.isfinite(t) & np.isfinite(pos)
dfc = pd.DataFrame({"t": t, "p": pos})[finite].copy()

# 時間でソート & 同一時刻は平均
dfc = dfc.groupby("t", as_index=False).mean().sort_values("t")

# 時間スパン適用（片方だけ指定でもOK）
tmin, tmax = dfc["t"].min(), dfc["t"].max()
t0 = tmin if T_START is None else T_START
t1 = tmax if T_END   is None else T_END
if t0 > t1:
    t0, t1 = t1, t0
dfc = dfc[(dfc["t"] >= t0) & (dfc["t"] <= t1)]

# 配列化
t = dfc["t"].to_numpy()
p = dfc["p"].to_numpy()
if len(t) < 3:
    raise ValueError("この時間範囲では点が少なすぎます。T_START/T_END を調整してください。")

# --- 速度・加速度（選んだ軸のみ） ---
pdot  = np.gradient(p, t)   # 速度（mm/s）
pddot = np.gradient(pdot, t)  # 加速度（mm/s^2）

# （任意）ノイズが強いときの移動平均
# def smooth(a, w=5): return np.convolve(a, np.ones(w)/w, mode="same")
# pdot  = smooth(pdot, 5)
# pddot = smooth(pddot, 5)

# --- 可視化（位置・速度・加速度の3段） ---
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.suptitle(f"{POINT} | {AXIS}-axis  [{t0:.2f}–{t1:.2f}s]")

axes[0].plot(t, p)
axes[0].set_ylabel(f"{AXIS} [mm]")
axes[0].grid(True)

axes[1].plot(t, pdot)
axes[1].axhline(0, lw=1)
axes[1].set_ylabel(f"d{AXIS}/dt [mm/s]")
axes[1].grid(True)

axes[2].plot(t, pddot)
axes[2].axhline(0, lw=1)
axes[2].set_ylabel(f"d²{AXIS}/dt² [mm/s²]")
axes[2].set_xlabel("time [s]")
axes[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
