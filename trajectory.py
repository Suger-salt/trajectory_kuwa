# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



### 軌跡を動画で確認したい
# ====== 設定（ここだけ触ればOK） ======
FILE = "2025_1025_dobusarai.csv"
POINT = "kuwa - 1"    # 例: "kuwa - 2"
T_START = 0.0         # Noneなら最小時刻
T_END = 25          # Noneなら最大時刻
# =====================================

# --- CSV読み込み ---
df = pd.read_csv(FILE)

# --- time / x / z を数値化 + 有限値だけに ---
time = pd.to_numeric(df["time"], errors="coerce")
x = pd.to_numeric(df[f"{POINT}_x"], errors="coerce")
z = pd.to_numeric(df[f"{POINT}_z"], errors="coerce")
mask = np.isfinite(time) & np.isfinite(x) & np.isfinite(z)

# --- 時間範囲フィルタ（片方だけ指定でもOK） ---
if (T_START is not None) or (T_END is not None):
    t0 = T_START if T_START is not None else np.nanmin(time)
    t1 = T_END   if T_END   is not None else np.nanmax(time)
    if t0 > t1:  # 念のため入れ替え
        t0, t1 = t1, t0
    mask &= (time >= t0) & (time <= t1)

# --- 抽出配列 ---
t_sel = time[mask].to_numpy()
x_sel = x[mask].to_numpy()
z_sel = z[mask].to_numpy()
if len(t_sel) < 2:
    raise ValueError("選択された時間範囲に十分なデータがありません。T_START/T_END を見直してください。")

# --- 図の用意（XZ） ---
fig, ax = plt.subplots(figsize=(6, 5))
(line,) = ax.plot([], [], lw=2)        # 軌跡（線）
(point,) = ax.plot([], [], "o")        # 現在位置（点）
ax.set_xlabel("X [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title(f"{POINT} (XZ)")
ax.grid(True)

# 軸範囲（少しマージン）
xm, xM = x_sel.min(), x_sel.max()
zm, zM = z_sel.min(), z_sel.max()
xr = xM - xm if xM > xm else 1.0
zr = zM - zm if zM > zm else 1.0
ax.set_xlim(xm - 0.05*xr, xM + 0.05*xr)
ax.set_ylim(zm - 0.05*zr, zM + 0.05*zr)

# 時刻表示（右上）
time_text = ax.text(0.98, 0.02, "", ha="right", va="bottom", transform=ax.transAxes)

# --- アニメ関数 ---
def init():
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text("")
    return line, point, time_text

def update(frame):
    k = frame + 1
    line.set_data(x_sel[:k], z_sel[:k])
    point.set_data([x_sel[frame]], [z_sel[frame]])  # 単一点は配列で渡す
    time_text.set_text(f"t = {t_sel[frame]:.2f} s")
    return line, point, time_text

# フレーム間隔（データのサンプリング間隔から自動計算、fallback=10ms）
dt_ms = 10
if len(t_sel) >= 2:
    dt = float(t_sel[1] - t_sel[0])
    if np.isfinite(dt) and dt > 0:
        dt_ms = int(round(1000 * dt))

ani = animation.FuncAnimation(
    fig, update, frames=len(t_sel), init_func=init, interval=dt_ms, blit=True
)

plt.tight_layout()
plt.show()
