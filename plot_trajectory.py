# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUTDIR = "trajectory_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- ユーティリティ ----------
def ensure_numeric(df, cols):
    """指定列を数値化（失敗は NaN）"""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def finite_xyz(df, base, clip_outliers=False):
    """x/y/z を数値化→有限値のみ抽出→(任意)外れ値クリップ"""
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"列がありません: {c}\n利用可能: {df.columns.tolist()}")

    df = ensure_numeric(df.copy(), cols)
    x = df[cols[0]].to_numpy()
    y = df[cols[1]].to_numpy()
    z = df[cols[2]].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    if x.size == 0:
        raise ValueError("有限な座標がありません（NaN/Infしかない可能性）。")

    if clip_outliers:
        # 1–99パーセンタイルで軽くクリップ（必要に応じてON）
        for arr in (x, y, z):
            lo, hi = np.percentile(arr, [1, 99])
            np.clip(arr, lo, hi, out=arr)

    return x, y, z

def safe_equal_limits_2d(ax, a, b, margin=0.05):
    """2D：等倍軸＋NaN/Inf対策"""
    a_min, a_max = np.nanmin(a), np.nanmax(a)
    b_min, b_max = np.nanmin(b), np.nanmax(b)
    if not np.isfinite([a_min, a_max, b_min, b_max]).all():
        ax.autoscale()
        return
    ar = a_max - a_min or 1.0
    br = b_max - b_min or 1.0
    ax.set_xlim(a_min - margin*ar, a_max + margin*ar)
    ax.set_ylim(b_min - margin*br, b_max + margin*br)
    ax.set_aspect("equal", adjustable="box")

def safe_equal_limits_3d(ax, x, y, z, margin=0.05):
    """3D：立方体等倍＋NaN/Inf対策"""
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
        ax.autoscale()
        return
    xr, yr, zr = (xmax-xmin or 1.0), (ymax-ymin or 1.0), (zmax-zmin or 1.0)
    maxr = max(xr, yr, zr)
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    half = (1 + margin) * maxr / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

def plot_2d(a, b, xlabel, ylabel, title, savepath):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(a, b)  # 色指定なし（デフォルト）
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    safe_equal_limits_2d(ax, a, b)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()

def plot_3d(x, y, z, title, savepath):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title)
    safe_equal_limits_3d(ax, x, y, z)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.show()

# ---------- メイン処理（関数引数で時間範囲を制御） ----------
def main(t_start=None, t_end=None, point="kuwa - 1", csv_path="2025_1025_dobusarai.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- 時間フィルタ（start / end の片方だけでもOK） ---
    if "time" in df.columns:
        tmin, tmax, n = df["time"].min(), df["time"].max(), len(df)
        print(f"Time range (raw): {tmin} ~ {tmax} (len={n})")

        # 片方だけ指定されても動くように補完
        if (t_start is not None) or (t_end is not None):
            t0 = t_start if t_start is not None else tmin
            t1 = t_end if t_end is not None else tmax
            if t0 > t1:  # 安全のため順番を正す
                t0, t1 = t1, t0
            df = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
            print(f"Filtered: {df['time'].min()} ~ {df['time'].max()} (len={len(df)})")
    else:
        if (t_start is not None) or (t_end is not None):
            print("⚠ time列がないため、時間フィルタは無視されます。")




    # 座標抽出（NaN/Inf対策込み）
    x, y, z = finite_xyz(df, point, clip_outliers=False)
    base = f"{point.replace(' ', '_')}_{os.path.basename(csv_path)}"
    if (t_start is not None) and (t_end is not None):
        base = f"{point.replace(' ', '_')}_{t_start}-{t_end}_{os.path.basename(csv_path)}"

    title_base = f"{point} | {os.path.basename(csv_path)}"
    if (t_start is not None) and (t_end is not None):
        title_base += f" | {t_start}–{t_end}s"

    # XY
    plot_2d(
        x, y, "X [mm]", "Y [mm]",
        title_base + " (XY)",
        os.path.join(OUTDIR, f"{base}_XY.png")
    )
    # XZ
    plot_2d(
        x, z, "X [mm]", "Z [mm]",
        title_base + " (XZ)",
        os.path.join(OUTDIR, f"{base}_XZ.png")
    )
    # YZ
    plot_2d(
        y, z, "Y [mm]", "Z [mm]",
        title_base + " (YZ)",
        os.path.join(OUTDIR, f"{base}_YZ.png")
    )
    # 3D
    plot_3d(
        x, y, z,
        title_base + " (3D)",
        os.path.join(OUTDIR, f"{base}_3D.png")
    )

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot trajectory for a single point (static).")
    parser.add_argument("--file", type=str, default="2025_1025_dobusarai.csv", help="CSV file path")
    parser.add_argument("--point", type=str, default="kuwa - 1", help="Point name, e.g., 'kuwa - 1'")
    parser.add_argument("--start", type=float, default=None, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="End time (seconds)")
    args = parser.parse_args()

    # 受け取り確認（デバッグ）
    print(f"[DEBUG] received: t_start={args.start}, t_end={args.end}, point={args.point}, file={args.file}")

    # 実行（★呼び出しをコメントアウトしない）
    main(t_start=args.start, t_end=args.end, point=args.point, csv_path=args.file)

