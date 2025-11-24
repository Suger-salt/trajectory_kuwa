import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 改良版 DMPクラス (幅の計算をロバストにしました)
# ==========================================
class DMP:
    def __init__(self, n_bfs=100, alpha=25.0, beta=6.25):
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta
        self.w = None

    def fit(self, path, dt):
        n_steps, n_dims = path.shape
        self.n_dims = n_dims
        self.y0 = path[0]
        self.g = path[-1]

        # 速度・加速度
        x = path
        dx = np.gradient(x, dt, axis=0)
        ddx = np.gradient(dx, dt, axis=0)

        tau = n_steps * dt
        t = np.linspace(0, tau, n_steps)
        s = np.exp(-self.alpha * t / tau)

        # --- 【修正ポイント】基底関数の配置と幅の計算 ---
        # 時間軸に対して均等に配置するのではなく、位相変数sに対して配置
        self.centers = np.exp(-self.alpha * np.linspace(0, 1, self.n_bfs))

        # 隣のセンターとの距離に応じて幅(h)を決める（これで数が多くても隙間ができない）
        self.widths = np.zeros(self.n_bfs)
        for i in range(self.n_bfs - 1):
            # 隣のセンターとの距離の二乗に反比例させる
            self.widths[i] = 1.0 / ((self.centers[i + 1] - self.centers[i]) ** 2)
        self.widths[-1] = self.widths[-2]  # 最後は一つ前と同じにする

        # 重み学習
        self.w = np.zeros((n_dims, self.n_bfs))
        for d in range(n_dims):
            g_d = self.g[d]
            y0_d = self.y0[d]
            scale = g_d - y0_d
            # スケールが小さすぎる場合の安定化
            if abs(scale) < 1e-4:
                scale = 1e-4

            K = self.alpha * self.beta
            D = self.alpha
            f_target = (
                tau**2 * ddx[:, d] + D * tau * dx[:, d] + K * (x[:, d] - g_d)
            ) / scale

            for i in range(self.n_bfs):
                psi = np.exp(-self.widths[i] * (s - self.centers[i]) ** 2)
                # 活性度が低すぎるデータは無視する（ゼロ除算防止）
                weight_val = np.sum(s * psi * f_target)
                activation = np.sum(s**2 * psi)

                if activation > 1e-10:
                    self.w[d, i] = weight_val / activation
                else:
                    self.w[d, i] = 0

    def step(self, y, dy, s, tau, g, y0):
        # 実行時も同じ幅・中心を使う
        psi = np.exp(-self.widths * (s - self.centers) ** 2)

        # 加重平均
        sum_psi = np.sum(psi)
        if sum_psi < 1e-10:
            f_val = np.zeros(self.n_dims)  # 活性化していないときは力ゼロ
        else:
            f_val = np.sum(self.w * psi, axis=1) / sum_psi * s

        scale = g - y0
        # DMPの運動方程式
        ddy = (self.alpha * (self.beta * (g - y) - tau * dy) + scale * f_val) / (tau**2)
        return ddy

    def rollout(self, y0, g, tau, dt):
        n_steps = int(tau / dt)
        t_arr = np.linspace(0, tau, n_steps)
        y_track = np.zeros((n_steps, self.n_dims))
        y = y0.copy()
        dy = np.zeros(self.n_dims)
        s = 1.0

        for i in range(n_steps):
            y_track[i] = y
            ddy = self.step(y, dy, s, tau, g, y0)
            dy += ddy * dt
            y += dy * dt
            ds = -self.alpha * s / tau
            s += ds * dt
        return y_track, t_arr


# ==========================================
# メイン実行部
# ==========================================
if __name__ == "__main__":
    file_path = "./csv_segment/kuwa_segment_output_02_11.0s-17.0s.csv"
    df = pd.read_csv(file_path)

    # データ準備
    target_cols = ["kuwa - 1_x", "kuwa - 1_y", "kuwa - 1_z"]
    trajectory_raw = (
        df[target_cols].interpolate(method="linear", limit_direction="both").values
    )
    time_raw = df["time"].values
    trajectory_smooth = (
        pd.DataFrame(trajectory_raw)
        .rolling(window=10, center=True, min_periods=1)
        .mean()
        .values
    )

    dt = time_raw[1] - time_raw[0]
    if np.isnan(dt) or dt == 0:
        dt = 0.01

    # ★ここで数を変えても大丈夫になります
    n_bfs_test = 100

    # print(f"DMP Training with {n_bfs_test} BFs...")
    # dmp = DMP(n_bfs=n_bfs_test)
    # dmp.fit(trajectory_smooth, dt)

    # 1. パディングデータの作成 (学習の前にやる！)
    # 最後の座標を500個(約5秒分)コピーして後ろにくっつける
    padding = np.tile(trajectory_smooth[-1], (500, 1))
    trajectory_padded = np.vstack([trajectory_smooth, padding])

    # 2. DMPの学習 (★ここでパディングしたデータを使う！)
    print(f"DMP Training with {n_bfs_test} BFs...")
    dmp = DMP(n_bfs=n_bfs_test)
    dmp.fit(trajectory_padded, dt)  # <--- smooth ではなく padded を渡す

    # 3. 再生設定
    # 時間(tau)も「パディングを含んだ長さ」にする必要があります
    tau_padded = len(trajectory_padded) * dt
    y0 = trajectory_padded[0]
    g = trajectory_padded[-1]

    # 4. 軌道生成 (Rollout)
    # 全体の軌道が出てきます
    y_repro, t_repro = dmp.rollout(y0, g, tau_padded, dt)

    # 5. 結果の切り出し (プロット用)
    # グラフで見るときは、元のデータの長さ分だけ取り出すと綺麗です
    original_len = len(trajectory_smooth)
    y_repro = y_repro[:original_len]
    t_repro = t_repro[:original_len]

    # グラフ描画
    fig = plt.figure(figsize=(12, 6))  # サイズを少し大きくしました

    # --- 3Dプロット ---
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(
        trajectory_smooth[:, 0],
        trajectory_smooth[:, 1],
        trajectory_smooth[:, 2],
        "k--",
        label="Original (Human)",
        alpha=0.3,
    )
    ax1.plot(
        y_repro[:, 0],
        y_repro[:, 1],
        y_repro[:, 2],
        "b-",
        linewidth=2,
        label="DMP Reproduction",
    )

    # ★ここに単位を追加しました
    ax1.set_title(f"3D Trajectory (n_bfs={n_bfs_test})")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_zlabel("Z [mm]")
    ax1.legend()

    # --- 時系列プロット ---
    ax2 = fig.add_subplot(122)
    labels = ["X", "Y", "Z"]
    colors = ["r", "g", "b"]  # 色分け (赤:X, 緑:Y, 青:Z)

    for i in range(3):
        ax2.plot(
            time_raw,
            trajectory_smooth[:, i],
            linestyle="--",
            color=colors[i],
            alpha=0.3,
        )
        ax2.plot(
            time_raw[0] + t_repro,
            y_repro[:, i],
            linestyle="-",
            color=colors[i],
            label=f"{labels[i]}",
        )

    ax2.set_title("Time Series")
    ax2.set_xlabel("Time [s]")
    # ★ここにも単位を追加
    ax2.set_ylabel("Position [mm]")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
