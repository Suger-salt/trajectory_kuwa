import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 1. DMPクラス
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
        x = path
        dx = np.gradient(x, dt, axis=0)
        ddx = np.gradient(dx, dt, axis=0)
        tau = n_steps * dt
        t = np.linspace(0, tau, n_steps)
        s = np.exp(-self.alpha * t / tau)
        self.centers = np.exp(-self.alpha * np.linspace(0, 1, self.n_bfs))
        self.widths = np.zeros(self.n_bfs)
        for i in range(self.n_bfs - 1):
            self.widths[i] = 1.0 / ((self.centers[i + 1] - self.centers[i]) ** 2)
        self.widths[-1] = self.widths[-2]
        self.w = np.zeros((n_dims, self.n_bfs))
        for d in range(n_dims):
            g_d = self.g[d]
            y0_d = self.y0[d]
            scale = g_d - y0_d
            if abs(scale) < 1e-4:
                scale = 1e-4
            K = self.alpha * self.beta
            D = self.alpha
            f_target = (
                tau**2 * ddx[:, d] + D * tau * dx[:, d] + K * (x[:, d] - g_d)
            ) / scale
            for i in range(self.n_bfs):
                psi = np.exp(-self.widths[i] * (s - self.centers[i]) ** 2)
                activation = np.sum(s**2 * psi)
                self.w[d, i] = (
                    np.sum(s * psi * f_target) / activation if activation > 1e-10 else 0
                )

    def step(self, y, dy, s, tau, g, y0):
        psi = np.exp(-self.widths * (s - self.centers) ** 2)
        sum_psi = np.sum(psi)
        if sum_psi > 1e-10:
            f_val = np.sum(self.w * psi, axis=1) / sum_psi * s
        else:
            f_val = np.zeros(self.n_dims)
        scale = g - y0
        ddy = (self.alpha * (self.beta * (g - y) - tau * dy) + scale * f_val) / (tau**2)
        return ddy

    def rollout(self, y0, g, tau, dt):
        n_steps = int(tau / dt)
        y_track = np.zeros((n_steps, self.n_dims))
        y = y0.copy()
        dy = np.zeros(self.n_dims)
        s = 1.0
        for i in range(n_steps):
            y_track[i] = y
            ddy = self.step(y, dy, s, tau, g, y0)
            dy += ddy * dt
            y += dy * dt
            s += (-self.alpha * s / tau) * dt
        return y_track


# ==========================================
# 2. 3リンクアームクラス
# ==========================================
class ThreeLinkArm_3D_TipFollow:
    def __init__(self, l1=1200.0, l2=1200.0, l_kuwa=400.0, origin=(0, 0, 0)):
        self.l1 = l1
        self.l2 = l2
        self.l_kuwa = l_kuwa
        self.origin = np.array(origin)

    def inverse_kinematics_tip(self, tx, ty, tz, target_phi):
        dx = tx - self.origin[0]
        dy = ty - self.origin[1]
        theta_base = np.arctan2(dy, dx)
        r_tip = np.sqrt(dx**2 + dy**2)
        z_tip = tz - self.origin[2]
        r_wrist = r_tip - self.l_kuwa * np.cos(target_phi)
        z_wrist = z_tip - self.l_kuwa * np.sin(target_phi)
        dist_sq = r_wrist**2 + z_wrist**2
        max_dist = self.l1 + self.l2
        if dist_sq > max_dist**2:
            scale = max_dist / np.sqrt(dist_sq)
            r_wrist *= scale
            z_wrist *= scale
            dist_sq = r_wrist**2 + z_wrist**2
        cos_th2 = (dist_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_th2 = np.clip(cos_th2, -1.0, 1.0)
        theta2 = -np.arccos(cos_th2)  # Elbow Up
        k1 = self.l1 + self.l2 * np.cos(theta2)
        k2 = self.l2 * np.sin(theta2)
        theta1 = np.arctan2(z_wrist, r_wrist) - np.arctan2(k2, k1)
        return theta_base, theta1, theta2, target_phi

    def get_joint_positions(self, theta_base, theta1, theta2, phi):
        r0, z0 = 0, 0
        r1 = self.l1 * np.cos(theta1)
        z1 = self.l1 * np.sin(theta1)
        r2 = r1 + self.l2 * np.cos(theta1 + theta2)
        z2 = z1 + self.l2 * np.sin(theta1 + theta2)
        r3 = r2 + self.l_kuwa * np.cos(phi)
        z3 = z2 + self.l_kuwa * np.sin(phi)

        def to_3d(r, z):
            x = self.origin[0] + r * np.cos(theta_base)
            y = self.origin[1] + r * np.sin(theta_base)
            z = self.origin[2] + z
            return np.array([x, y, z])

        p0 = to_3d(r0, z0)
        p1 = to_3d(r1, z1)
        p2 = to_3d(r2, z2)
        p3 = to_3d(r3, z3)
        return np.array([p0, p1, p2, p3])


# ==========================================
# メイン実行部
# ==========================================
if __name__ == "__main__":
    file_path = "./csv_segment/kuwa_segment_output_02_11.0s-17.0s.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("CSVファイルが見つかりません。")
        exit()

    # データ処理
    p_wrist = df[["kuwa - 1_x", "kuwa - 1_y", "kuwa - 1_z"]].interpolate().values
    p2 = df[["kuwa - 2_x", "kuwa - 2_y", "kuwa - 2_z"]].interpolate().values
    p3 = df[["kuwa - 3_x", "kuwa - 3_y", "kuwa - 3_z"]].interpolate().values
    p_tip = (p2 + p3) / 2.0

    vec = p_tip - p_wrist
    base_pos = np.array([-1000, -500, 0])
    base_to_wrist = p_wrist[:, :2] - base_pos[:2]
    dist = np.linalg.norm(base_to_wrist, axis=1)
    dir_x = base_to_wrist[:, 0] / dist
    dir_y = base_to_wrist[:, 1] / dist
    vec_r = vec[:, 0] * dir_x + vec[:, 1] * dir_y
    vec_z = vec[:, 2]
    phi_raw = np.arctan2(vec_z, vec_r)

    # 角度補正
    phi_raw = np.where(np.abs(phi_raw) < np.pi / 2, phi_raw - np.pi, phi_raw)
    SAFE_ANGLE_MAX = np.deg2rad(-20)
    phi_raw = np.minimum(phi_raw, SAFE_ANGLE_MAX)

    # ★学習データの前処理: Z < 0 のデータを 0 に強制補正する
    # これでDMP自体が「地面より下」を学習しないようにする
    traj_raw_pos = p_tip.copy()
    traj_raw_pos[:, 2] = np.maximum(traj_raw_pos[:, 2], 0)  # Z >= 0

    # 位置と角度を結合
    traj_combined = np.column_stack([traj_raw_pos, phi_raw])

    time_raw = df["time"].values
    dt = time_raw[1] - time_raw[0] if (len(time_raw) > 1) else 0.01

    traj_smooth = (
        pd.DataFrame(traj_combined)
        .rolling(10, min_periods=1, center=True)
        .mean()
        .values
    )

    # ★パディング (最後の方で浮き上がらないよう、Z=0付近なら0で埋める)
    last_pose = traj_smooth[-1].copy()
    if last_pose[2] < 10:
        last_pose[2] = 0  # ほぼ地面なら着地させる
    padding = np.tile(last_pose, (300, 1))
    traj_padded = np.vstack([traj_smooth, padding])

    print("DMP Training...")
    dmp = DMP(n_bfs=100)
    dmp.fit(traj_padded, dt)

    tau = len(traj_padded) * dt
    repro_padded = dmp.rollout(traj_padded[0], traj_padded[-1], tau, dt)
    repro_play = repro_padded[: len(traj_smooth)]

    y_play = repro_play[:, :3]
    phi_play = repro_play[:, 3]

    # --- シミュレーション ---
    arm_3d = ThreeLinkArm_3D_TipFollow(
        l1=900.0, l2=800.0, l_kuwa=150.0, origin=(-750, 0, 750)
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Visual Servoing Simulation")
    ax.set_xlim(-1500, 500)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-500, 1500)

    xx, yy = np.meshgrid(np.linspace(-1500, 500, 10), np.linspace(-1000, 1000, 10))
    ax.plot_wireframe(xx, yy, np.zeros_like(xx), color="gray", alpha=0.2)
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="red", alpha=0.2)

    (line_arm,) = ax.plot([], [], [], "o-", lw=2, color="orange", label="Arm")
    (line_kuwa,) = ax.plot(
        [], [], [], "-", lw=4, color="brown", label="Handle"
    )  # 柄を少し太く

    # ★刃を2つのパーツで描画 (面とエッジ)
    (line_blade_face,) = ax.plot(
        [], [], [], "-", lw=1, color="gray", alpha=0.5
    )  # 面（細い線）
    (line_blade_edge,) = ax.plot(
        [], [], [], "-", lw=5, color="red", label="Edge"
    )  # 刃先（太い赤）

    (line_traj,) = ax.plot(y_play[:, 0], y_play[:, 1], y_play[:, 2], "b:", alpha=0.3)
    ax.plot([arm_3d.origin[0]], [arm_3d.origin[1]], [arm_3d.origin[2]], "kx", ms=10)
    ax.legend()

    def update(frame):
        # 1. ターゲット（DMPの計算結果）を取得
        target = y_play[frame].copy()

        # 地面めり込み防止ガード
        if target[2] < 0:
            target[2] = 0

        current_phi = phi_play[frame]

        # 2. ★ここが抜けていました！★
        # 逆運動学(IK)を解いて、関節位置(joints)を計算する
        tb, t1, t2, phi = arm_3d.inverse_kinematics_tip(
            target[0], target[1], target[2], current_phi
        )
        joints = arm_3d.get_joint_positions(tb, t1, t2, phi)

        # 3. アームと柄の描画更新
        # アーム (Base -> Shoulder -> Elbow -> Wrist)
        line_arm.set_data(joints[:3, 0], joints[:3, 1])
        line_arm.set_3d_properties(joints[:3, 2])

        # クワの柄 (Wrist -> Tip)
        line_kuwa.set_data(joints[2:, 0], joints[2:, 1])
        line_kuwa.set_3d_properties(joints[2:, 2])

        # 4. 刃（ブレード）の計算と描画
        tip_pos = joints[3]
        wrist_pos = joints[2]
        vec_handle = tip_pos - wrist_pos
        vec_up = np.array([0, 0, 1])

        # 外積で「横方向」を出す
        vec_blade_dir = np.cross(vec_handle, vec_up)
        norm = np.linalg.norm(vec_blade_dir)

        # 刃の幅 (片側 150mm)
        width = 150.0
        if norm > 1e-6:
            vec_blade_dir = vec_blade_dir / norm * width
        else:
            vec_blade_dir = np.array([width, 0, 0])

        # 3つの頂点を計算
        blade_left = tip_pos + vec_blade_dir
        blade_right = tip_pos - vec_blade_dir
        blade_root = tip_pos - (vec_handle * 0.4)  # 柄の少し上

        # 三角形を一筆書き
        tri_x = [blade_left[0], blade_right[0], blade_root[0], blade_left[0]]
        tri_y = [blade_left[1], blade_right[1], blade_root[1], blade_left[1]]
        tri_z = [blade_left[2], blade_right[2], blade_root[2], blade_left[2]]

        # 面（グレーの細い線）
        face_x = [blade_root[0], tip_pos[0]]
        face_y = [blade_root[1], tip_pos[1]]
        face_z = [blade_root[2], tip_pos[2]]

        # 描画オブジェクトにセット
        line_blade_face.set_data(face_x, face_y)
        line_blade_face.set_3d_properties(face_z)

        line_blade_edge.set_data(tri_x, tri_y)
        line_blade_edge.set_3d_properties(tri_z)

        return line_arm, line_kuwa, line_blade_face, line_blade_edge

    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(y_play), 1), interval=50, blit=False
    )
    plt.show()
