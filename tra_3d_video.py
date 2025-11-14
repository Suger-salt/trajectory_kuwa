# tra_3d_video.py
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

def load_point(csv_path, point, t_start=None, t_end=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # 必須列チェック
    for col in ["time", f"{point}_x", f"{point}_y", f"{point}_z"]:
        if col not in df.columns:
            raise KeyError(f"列がありません: {col}")

    # 数値化 & 有限値
    t = pd.to_numeric(df["time"], errors="coerce")
    x = pd.to_numeric(df[f"{point}_x"], errors="coerce")
    y = pd.to_numeric(df[f"{point}_y"], errors="coerce")
    z = pd.to_numeric(df[f"{point}_z"], errors="coerce")
    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    dfc = pd.DataFrame({"t": t, "x": x, "y": y, "z": z})[finite].copy()

    # 時間で集計（重複時刻があれば平均）→ソート
    dfc = dfc.groupby("t", as_index=False).mean().sort_values("t")

    # スパン適用（片方だけでもOK）
    if (t_start is not None) or (t_end is not None):
        tmin, tmax = dfc["t"].min(), dfc["t"].max()
        t0 = tmin if t_start is None else float(t_start)
        t1 = tmax if t_end   is None else float(t_end)
        if t0 > t1: t0, t1 = t1, t0
        dfc = dfc[(dfc["t"] >= t0) & (dfc["t"] <= t1)]

    if len(dfc) < 2:
        raise ValueError("アニメーションに十分な点がありません（時間範囲を見直してください）。")

    t = dfc["t"].to_numpy()
    x = dfc["x"].to_numpy()
    y = dfc["y"].to_numpy()
    z = dfc["z"].to_numpy()
    return t, x, y, z

def speed_from_xyz(t, x, y, z):
    # 不等間隔にも対応
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return v

def make_segments(x, y, z):
    """(N,3) -> (N-1, 2, 3) 連続線分の集合"""
    pts = np.column_stack([x, y, z])
    return np.stack([pts[:-1], pts[1:]], axis=1)

def build_3d(ax, x, y, z, v, cmap, vmin=None, vmax=None):
    # カラー正規化（未指定なら外れ値耐性のため分位点を採用）
    if vmin is None or vmax is None:
        lo, hi = np.percentile(v, [5, 95])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
        if vmax <= vmin:
            vmin, vmax = float(np.min(v)), float(np.max(v)) + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax)
    segs = make_segments(x, y, z)
    # 線分ごとの速度値（中点近似）
    v_mid = (v[:-1] + v[1:]) * 0.5

    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(v_mid)
    h_line = ax.add_collection3d(lc)

    # 現在位置の点
    h_point = ax.plot([x[0]], [y[0]], [z[0]], "o")[0]

    # カラーバー
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(v)  # for colorbar
    cb = plt.colorbar(mappable, ax=ax, pad=0.02, shrink=0.7)
    cb.set_label("speed [mm/s]")

    # 軸範囲（立方体で等倍率）
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    maxr = max(xmax-xmin, ymax-ymin, zmax-zmin)
    half = maxr * 0.55 if maxr > 0 else 1.0
    ax.set_xlim(cx-half, cx+half)
    ax.set_ylim(cy-half, cy+half)
    ax.set_zlim(cz-half, cz+half)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    return h_line, h_point, norm

def animate(
    csv_path, point, t_start, t_end,
    mode, tail_seconds, cmap, vmin, vmax,
    fps, speed_factor, out
):
    t, x, y, z = load_point(csv_path, point, t_start, t_end)
    v = speed_from_xyz(t, x, y, z)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    title_txt = ax.set_title(f"{point} | {os.path.basename(csv_path)}", pad=12)
    time_txt = fig.text(0.5, 0.96, "", ha="center", va="top")

    h_line, h_point, norm = build_3d(ax, x, y, z, v, cmap, vmin, vmax)

    # 再生間隔(ms)
    if len(t) >= 2:
        dt_med = np.median(np.diff(t))
        interval_ms = max(1, int(round(1000 * dt_med / max(1e-6, speed_factor))))
    else:
        interval_ms = 10

    def set_tail(k):
        """現在フレームkに対して表示区間の開始indexを返す"""
        if mode == "persistent":
            return 0
        elif mode == "tail":
            t_start_tail = t[k] - tail_seconds
            # 最初に tail 開始時刻以上になるindex
            i0 = np.searchsorted(t, t_start_tail, side="left")
            return max(0, min(i0, k))
        else:
            return 0

    def update(frame):
        k = frame
        i0 = set_tail(k)
        # 線分と色を更新
        segs = make_segments(x[i0:k+1], y[i0:k+1], z[i0:k+1])
        if len(segs) == 0:
            # 1点だけのときは空線でしのぐ
            h_line.set_segments([])
        else:
            h_line.set_segments(segs)
            v_mid = (v[i0:k] + v[i0+1:k+1]) * 0.5
            h_line.set_array(v_mid)

        # 現在点
        h_point.set_data_3d([x[k]], [y[k]], [z[k]])

        # 時刻表示
        time_txt.set_text(f"t = {t[k]:.2f} s   (mode: {mode}"
                          + (f", tail={tail_seconds:.1f}s" if mode == "tail" else "")
                          + ")")
        return h_line, h_point, time_txt

    ani = animation.FuncAnimation(
        fig, update, frames=len(t), interval=interval_ms, blit=False
    )

    # 保存（任意）
    if out:
        ext = os.path.splitext(out)[1].lower()
        try:
            if ext == ".mp4":
                ani.save(out, writer="ffmpeg", fps=fps, dpi=200)
            elif ext in (".gif", ".apng"):
                writer = "pillow" if ext == ".gif" else "pillow"
                ani.save(out, writer=writer, fps=fps)
            else:
                raise ValueError("拡張子は .mp4 / .gif / .apng を推奨します。")
            print(f"✅ Saved: {out}")
        except Exception as e:
            print(f"⚠ 保存に失敗: {e}\n（ffmpeg未導入なら .gif での保存を試してください）")

    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="3D trajectory player with speed-colored trail")
    ap.add_argument("--file", type=str, default="2025_1025_dobusarai.csv")
    ap.add_argument("--point", type=str, default="kuwa - 1")
    ap.add_argument("--start", type=float, default=None, help="start time [s]")
    ap.add_argument("--end", type=float, default=None, help="end time [s]")
    ap.add_argument("--mode", choices=["persistent", "tail"], default="persistent",
                    help="trail mode: persistent or rolling tail")
    ap.add_argument("--tail-seconds", type=float, default=5.0,
                    help="when mode=tail, keep only the last N seconds")
    ap.add_argument("--cmap", type=str, default="plasma", help="matplotlib colormap")
    ap.add_argument("--vmin", type=float, default=None, help="speed color lower bound")
    ap.add_argument("--vmax", type=float, default=None, help="speed color upper bound")
    ap.add_argument("--fps", type=int, default=60, help="save fps if output")
    ap.add_argument("--speed-factor", type=float, default=1.0,
                    help=">1.0 for faster playback, <1.0 for slower")
    ap.add_argument("--out", type=str, default=None,
                    help="optional output path (.mp4/.gif/.apng). If omitted, just show.")
    args = ap.parse_args()

    animate(
        csv_path=args.file,
        point=args.point,
        t_start=args.start,
        t_end=args.end,
        mode=args.mode,
        tail_seconds=args.tail_seconds,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        fps=args.fps,
        speed_factor=args.speed_factor,
        out=args.out,
    )

if __name__ == "__main__":
    main()
