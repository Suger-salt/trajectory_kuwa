import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# 新しい関数を追加


def visualize_trajectory(poses, start_time, end_time, frame_step=1):
    """
    指定した時間区間内の代表点（P2）の軌跡を3Dで可視化する（静止画）。

    Parameters:
    poses (list): 全姿勢データ
    start_time (float): 開始時刻 (秒)
    end_time (float): 終了時刻 (秒)
    frame_step (int): 軌跡を間引く間隔 (1を指定すると全フレーム表示)
    """

    # 1. 時間指定に基づいたデータ範囲の特定 (create_animationのロジックを流用)
    if start_time is None:
        start_idx = 0
    else:
        start_idx = next((i for i, p in enumerate(poses) if p["time"] >= start_time), 0)

    if end_time is None:
        end_idx = len(poses) - 1
    else:
        end_idx = (
            next((i for i, p in enumerate(poses) if p["time"] > end_time), len(poses))
            - 1
        )

    if end_idx < start_idx:
        print(f"警告: 無効な時間範囲 ({start_time}s〜{end_time}s) です。")
        return None

    # データのスライス
    # end_idx + 1 にすることで終了フレームを含む
    subset_poses = poses[start_idx : end_idx + 1]

    if not subset_poses:
        print(
            f"警告: 指定された時間範囲 ({start_time}s〜{end_time}s) にデータがありません。"
        )
        return None

    # 2. P2の座標を抽出（軌跡の代表点）
    trajectory = []
    for i, pose in enumerate(subset_poses):
        if i % frame_step == 0:  # 間引き処理
            trajectory.append(pose["p2"])

    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 3. 軌跡のプロット（すべての点を結んで表示）
    # 軌跡の線
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        "r-",
        linewidth=2,
        alpha=0.7,
        label="P2 trajectory",
    )

    # 軌跡上の点
    ax.scatter(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c="r", marker="o", s=10
    )

    # 開始点と終了点を強調
    ax.scatter(*trajectory[0], c="green", s=100, label="start")
    ax.scatter(*trajectory[-1], c="blue", s=100, label="end")

    # 4. グラフ設定
    ax.set_title(f"P2 trajectory ({start_time:.2f}s - {end_time:.2f}s)", fontsize=14)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend()
    ax.grid(True)

    # 軸の範囲を軌跡全体が収まるように設定 (auto_scale_xyzの代わりにset_limを使用)

    # 軌跡の最小値と最大値を計算
    x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    z_min, z_max = trajectory[:, 2].min(), trajectory[:, 2].max()

    # 全軸の範囲を最も広い軸に合わせる (オプション: 見た目を立方体にするため)
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2
    center = trajectory.mean(axis=0)

    # 修正: set_xlim, set_ylim, set_zlim を使って範囲を直接設定
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    return fig


# 新しい関数を追加
def save_trajectory_segments(poses, time_segments, base_filename="kuwa_trajectory"):
    """
    時間区切り配列に基づき、軌跡の静止画を連続して保存する。
    """
    if len(time_segments) < 2:
        print("エラー: time_segments には2つ以上の時刻が必要です。")
        return

    # 保存フォルダの確認と作成
    output_directory = "trajectory"
    os.makedirs(output_directory, exist_ok=True)

    segment_pairs = list(zip(time_segments[:-1], time_segments[1:]))

    print(f"--- {len(segment_pairs)} 個の軌跡画像を保存します ---")

    for i, (start_time, end_time) in enumerate(segment_pairs):
        output_filename = (
            f"{base_filename}_{i+1:02d}_{start_time:.1f}s-{end_time:.1f}s.png"
        )
        output_path = os.path.join(output_directory, output_filename)

        print(
            f"\n[セグメント {i+1}] {start_time:.1f}s から {end_time:.1f}s までの軌跡を {output_path} に保存中..."
        )

        # 軌跡可視化関数を呼び出す
        fig = visualize_trajectory(poses, start_time, end_time)

        if fig:
            # PNGファイルとして保存
            fig.savefig(output_path)

            # --- 修正箇所：ループ内の処理を完結させる ---
            # 軌跡の確認をしたい場合は、ここで plt.show() して、ユーザーが閉じるのを待つ
            print(
                "保存完了。確認のため図を表示します。図を閉じると次のセグメントに進みます。"
            )
            plt.show()  # <--- 図を表示し、ユーザーが閉じるまで待機
            # plt.show() が実行されると、その図は自動で閉じられるため、plt.close(fig) は不要

    print("\nすべての軌跡画像の保存が完了しました。")


# CSVファイルを読み込む
def load_data(csv_path):
    """CSVファイルを読み込む"""
    df = pd.read_csv(csv_path)
    return df


# 3点から姿勢を計算
def calculate_pose(p2, p3, p5):
    """
    3点から鍬の姿勢（座標系とオイラー角）を計算

    Parameters:
    p2, p3, p5: 各点の座標 (x, y, z)

    Returns:
    dict: origin, x_axis, y_axis, z_axis, roll, pitch, yaw
    """
    # ベクトル計算
    v23 = p3 - p2  # p2からp3へのベクトル
    v25 = p5 - p2  # p2からp5へのベクトル

    # X軸: p2->p3方向
    x_axis = v23 / np.linalg.norm(v23)

    # Z軸: X軸とv25の外積
    z_axis_raw = np.cross(x_axis, v25)
    z_axis = z_axis_raw / np.linalg.norm(z_axis_raw)

    # Y軸: Z軸とX軸の外積
    y_axis = np.cross(z_axis, x_axis)

    # 回転行列を構築
    R = np.column_stack([x_axis, y_axis, z_axis])

    # オイラー角を計算 (ZYX順: Roll-Pitch-Yaw)
    roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) * 180 / np.pi
    yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

    return {
        "origin": p2,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "rotation_matrix": R,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


# 全フレームの姿勢を計算（5点すべて読み込む）
def calculate_all_poses(df):
    """全フレームの姿勢を計算"""
    poses = []

    for idx, row in df.iterrows():
        try:
            # 5点すべてを読み込む
            p1 = np.array([row["kuwa - 1_x"], row["kuwa - 1_y"], row["kuwa - 1_z"]])
            p2 = np.array([row["kuwa - 2_x"], row["kuwa - 2_y"], row["kuwa - 2_z"]])
            p3 = np.array([row["kuwa - 3_x"], row["kuwa - 3_y"], row["kuwa - 3_z"]])
            p4 = np.array([row["kuwa - 4_x"], row["kuwa - 4_y"], row["kuwa - 4_z"]])
            p5 = np.array([row["kuwa - 5_x"], row["kuwa - 5_y"], row["kuwa - 5_z"]])

            # NaNや無限大をチェック
            if (
                np.any(np.isnan(p1))
                or np.any(np.isnan(p2))
                or np.any(np.isnan(p3))
                or np.any(np.isnan(p4))
                or np.any(np.isnan(p5))
            ):
                print(f"警告: フレーム {idx} にNaNが含まれています。スキップします。")
                continue
            if (
                np.any(np.isinf(p1))
                or np.any(np.isinf(p2))
                or np.any(np.isinf(p3))
                or np.any(np.isinf(p4))
                or np.any(np.isinf(p5))
            ):
                print(
                    f"警告: フレーム {idx} に無限大が含まれています。スキップします。"
                )
                continue

            # 姿勢計算はp2, p3, p5を使用
            pose = calculate_pose(p2, p3, p5)

            # 計算結果もチェック
            if (
                np.isnan(pose["roll"])
                or np.isnan(pose["pitch"])
                or np.isnan(pose["yaw"])
            ):
                print(
                    f"警告: フレーム {idx} の姿勢計算結果にNaNが含まれています。スキップします。"
                )
                continue

            pose["time"] = row["time"]
            pose["frame_idx"] = idx
            # 5点すべてを保存
            pose["p1"] = p1
            pose["p2"] = p2
            pose["p3"] = p3
            pose["p4"] = p4
            pose["p5"] = p5
            poses.append(pose)
        except Exception as e:
            print(f"エラー: フレーム {idx} の処理中にエラーが発生しました: {e}")
            continue

    print(f"有効なフレーム数: {len(poses)} / {len(df)}")
    return poses


if __name__ == "__main__":

    # ----------------------------------------------------
    # 【必須の追加箇所】poses 変数の定義
    # ----------------------------------------------------
    # CSVファイルを読み込む
    csv_path = "2025_1025_dobusarai.csv"  # ← CSVファイルパスを指定
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(
            f"エラー: CSVファイル '{csv_path}' が見つかりません。パスを確認してください。"
        )
        # 処理を中断
        # ここではダミーデータなどを使わずに、エラーメッセージを出して終了するのが安全
        exit()

    # 全フレームの姿勢を計算
    poses = calculate_all_poses(df)
    print(f"計算完了: {len(poses)} フレーム")

    # データが存在しない場合のチェック
    if not poses:
        print("エラー: 有効な姿勢データが計算されませんでした。処理を中断します。")
        exit()

    # 軌跡を表示したい時間の区切りを指定 (例: 0s~5s, 5s~10s, 10s~15s)
    trajectory_times = [0.0, 5.0, 10.0, 15.0]

    save_trajectory_segments(
        poses, time_segments=trajectory_times, base_filename="P2_trajectory"
    )
