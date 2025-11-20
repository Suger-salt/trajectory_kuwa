import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


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


# 静止画で可視化（5点版）
def visualize_frame(pose, frame_idx=0):
    """特定フレームの姿勢を3Dで可視化"""
    fig = plt.figure(figsize=(14, 6))

    # 3Dプロット
    ax1 = fig.add_subplot(121, projection="3d")

    # 5点をプロット
    ax1.scatter(
        *pose["p1"],
        c="cyan",
        s=150,
        label="Point 1",
        edgecolors="black",
        linewidths=1.5,
    )
    ax1.scatter(
        *pose["p2"],
        c="red",
        s=150,
        label="Point 2 (原点)",
        edgecolors="black",
        linewidths=1.5,
    )
    ax1.scatter(
        *pose["p3"],
        c="green",
        s=150,
        label="Point 3",
        edgecolors="black",
        linewidths=1.5,
    )
    ax1.scatter(
        *pose["p4"],
        c="magenta",
        s=150,
        label="Point 4",
        edgecolors="black",
        linewidths=1.5,
    )
    ax1.scatter(
        *pose["p5"],
        c="blue",
        s=150,
        label="Point 5",
        edgecolors="black",
        linewidths=1.5,
    )

    # 鍬の形状（5点を結ぶ）
    points_line = np.array([pose["p1"], pose["p2"], pose["p3"], pose["p4"], pose["p5"]])
    ax1.plot(
        points_line[:, 0],
        points_line[:, 1],
        points_line[:, 2],
        "k-",
        linewidth=3,
        alpha=0.6,
    )

    # 参考：三角形も描画（p2-p3-p5）
    triangle = np.array([pose["p2"], pose["p3"], pose["p5"], pose["p2"]])
    ax1.plot(
        triangle[:, 0],
        triangle[:, 1],
        triangle[:, 2],
        "gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.3,
    )

    # 座標軸を描画
    axis_length = 100
    origin = pose["origin"]

    # X軸（赤）
    ax1.quiver(
        *origin,
        *pose["x_axis"] * axis_length,
        color="red",
        arrow_length_ratio=0.15,
        linewidth=3,
    )
    # Y軸（緑）
    ax1.quiver(
        *origin,
        *pose["y_axis"] * axis_length,
        color="green",
        arrow_length_ratio=0.15,
        linewidth=3,
    )
    # Z軸（青）
    ax1.quiver(
        *origin,
        *pose["z_axis"] * axis_length,
        color="blue",
        arrow_length_ratio=0.15,
        linewidth=3,
    )

    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_zlabel("Z", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.set_title(f"鍬の姿勢 (Frame {frame_idx})", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 軸の範囲を設定（5点すべてを含む）
    all_points = np.vstack([pose["p1"], pose["p2"], pose["p3"], pose["p4"], pose["p5"]])
    center = np.mean(all_points, axis=0)

    # 各軸の範囲を個別に計算
    x_range = np.ptp(all_points[:, 0])
    y_range = np.ptp(all_points[:, 1])
    z_range = np.ptp(all_points[:, 2])

    # 余裕を持たせる（固定値：各軸±1000で合計2000の範囲）
    x_margin = 1000
    y_margin = 1000
    z_margin = 1000

    ax1.set_xlim(center[0] - x_margin, center[0] + x_margin)
    ax1.set_ylim(center[1] - y_margin, center[1] + y_margin)
    ax1.set_zlim(center[2] - z_margin, center[2] + z_margin)

    # 姿勢角をテキスト表示
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    info_text = f"""
    時刻: {pose['time']:.3f} s
    
    姿勢角 (オイラー角):
    Roll:  {pose['roll']:.2f}°
    Pitch: {pose['pitch']:.2f}°
    Yaw:   {pose['yaw']:.2f}°
    
    原点位置 (Point 2):
    X: {pose['origin'][0]:.2f}
    Y: {pose['origin'][1]:.2f}
    Z: {pose['origin'][2]:.2f}
    
    各点の位置:
    P1: ({pose['p1'][0]:.1f}, {pose['p1'][1]:.1f}, {pose['p1'][2]:.1f})
    P2: ({pose['p2'][0]:.1f}, {pose['p2'][1]:.1f}, {pose['p2'][2]:.1f})
    P3: ({pose['p3'][0]:.1f}, {pose['p3'][1]:.1f}, {pose['p3'][2]:.1f})
    P4: ({pose['p4'][0]:.1f}, {pose['p4'][1]:.1f}, {pose['p4'][2]:.1f})
    P5: ({pose['p5'][0]:.1f}, {pose['p5'][1]:.1f}, {pose['p5'][2]:.1f})
    
    座標軸:
    X軸（赤）: p2→p3方向
    Y軸（緑）: 直交成分
    Z軸（青）: 外積で計算
    """
    ax2.text(
        0.1, 0.5, info_text, fontsize=11, verticalalignment="center", family="monospace"
    )

    plt.tight_layout()
    return fig


# 時系列グラフ
def plot_time_series(poses):
    """姿勢角の時系列変化をプロット"""
    times = [p["time"] for p in poses]
    rolls = [p["roll"] for p in poses]
    pitches = [p["pitch"] for p in poses]
    yaws = [p["yaw"] for p in poses]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(times, rolls, "r-", linewidth=2)
    axes[0].set_ylabel("Roll (°)")
    axes[0].grid(True)
    axes[0].set_title("kuwa_change_posture (Euler anges) ")

    axes[1].plot(times, pitches, "g-", linewidth=2)
    axes[1].set_ylabel("Pitch (°)")
    axes[1].grid(True)

    axes[2].plot(times, yaws, "b-", linewidth=2)
    axes[2].set_ylabel("Yaw (°)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)

    plt.tight_layout()
    return fig


# アニメーション作成（5点版）
def create_animation(
    poses, start_time=None, end_time=None, interval=50, save_path=None
):
    """姿勢変化のアニメーションを作成 (秒数指定対応)"""

    # --- 1. 時間指定に基づいたデータ範囲の特定 ---
    if start_time is not None:
        # 開始時刻 >= time の最初のフレームを検索
        # posesはtime順に並んでいる前提
        start_idx = next((i for i, p in enumerate(poses) if p["time"] >= start_time), 0)
    else:
        start_idx = 0

    if end_time is not None:
        # 終了時刻 <= time の最後のフレームを検索
        end_idx = (
            next((i for i, p in enumerate(poses) if p["time"] > end_time), len(poses))
            - 1
        )
    else:
        end_idx = len(poses) - 1

    # 範囲の検証と調整
    if start_idx >= len(poses) or end_idx < start_idx:
        print(
            f"エラー: 指定された時間範囲 ({start_time}s〜{end_time}s) に有効なデータがありません。"
        )
        return None

    # --- 2. データのスライス ---
    # Pythonのスライスは終点を含まないので、end_idxを終点として使うには +1 する
    subset_poses = poses[start_idx : end_idx + 1]

    # 既存のデータチェック
    if len(subset_poses) == 0:
        print("エラー: 指定された範囲に有効なデータがありません")
        return None

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 全体の表示範囲は元の poses 全体を使って計算を続行
    # ... (既存の表示範囲計算コードはそのまま)
    all_points = []
    for pose in poses:  # 範囲全体を使って計算（安定性のため）
        all_points.extend([pose["p1"], pose["p2"], pose["p3"], pose["p4"], pose["p5"]])
    all_points = np.array(all_points)

    # ... (NaNチェックとcenter, margin計算は省略せずに残す)
    valid_points = all_points[~np.isnan(all_points).any(axis=1)]
    center = np.mean(valid_points, axis=0)
    x_margin, y_margin, z_margin = 1000, 1000, 1000  # 固定マージン

    print(f"表示範囲: 中心={center}")
    print(f"X範囲: {center[0] - x_margin:.1f} ~ {center[0] + x_margin:.1f}")
    print(f"Y範囲: {center[1] - y_margin:.1f} ~ {center[1] + y_margin:.1f}")
    print(f"Z範囲: {center[2] - z_margin:.1f} ~ {center[2] + z_margin:.1f}")

    # --- 3. update関数の修正 (subset_posesを使用) ---
    def update(frame):
        ax.clear()
        # 修正点: frameは0から始まるため、subset_posesからデータを取得
        pose = subset_poses[frame]

        # ... (5点のプロット、鍬の形状、座標軸の描画コードは変更なし) ...
        ax.scatter(
            *pose["p1"],
            c="cyan",
            s=200,
            label="Point 1",
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            *pose["p2"],
            c="red",
            s=200,
            label="Point 2",
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            *pose["p3"],
            c="green",
            s=200,
            label="Point 3",
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            *pose["p4"],
            c="magenta",
            s=200,
            label="Point 4",
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            *pose["p5"],
            c="blue",
            s=200,
            label="Point 5",
            edgecolors="black",
            linewidths=2,
        )

        # 鍬の線を描画
        points_235_triangle = np.array(
            [pose["p2"], pose["p3"], pose["p5"], pose["p2"]]
        )  # 修正/確認

        # 描画コード：三角形
        ax.plot(
            points_235_triangle[:, 0],
            points_235_triangle[:, 1],
            points_235_triangle[:, 2],
            "k-",  # 色と線の種類はお好みで
            linewidth=3,
            alpha=0.8,
        )

        # 既存の線（例: 1-4や1-3を結ぶ線）はそのまま残す
        points_14 = np.array([pose["p1"], pose["p4"]])
        ax.plot(
            points_14[:, 0],
            points_14[:, 1],
            points_14[:, 2],
            "k-",
            linewidth=3,
            alpha=0.8,
        )

        points_13 = np.array([pose["p1"], pose["p3"]])
        ax.plot(
            points_13[:, 0],
            points_13[:, 1],
            points_13[:, 2],
            "k-",
            linewidth=3,
            alpha=0.8,
        )

        # 鍬の形状、座標軸の描画

        origin = pose["origin"]
        axis_length = 100
        ax.quiver(
            *origin,
            *pose["x_axis"] * axis_length,
            color="red",
            arrow_length_ratio=0.15,
            linewidth=3,
        )
        ax.quiver(
            *origin,
            *pose["y_axis"] * axis_length,
            color="green",
            arrow_length_ratio=0.15,
            linewidth=3,
        )
        ax.quiver(
            *origin,
            *pose["z_axis"] * axis_length,
            color="blue",
            arrow_length_ratio=0.15,
            linewidth=3,
        )

        # 各軸の範囲を個別に設定 (中略)
        ax.set_xlim(center[0] - x_margin, center[0] + x_margin)
        ax.set_ylim(center[1] - y_margin, center[1] + y_margin)
        ax.set_zlim(center[2] - z_margin, center[2] + z_margin)

        # 修正点: frame_idxはposeディクショナリに保存されている元のフレーム番号を使用
        frame_idx = pose["frame_idx"]
        ax.set_title(
            f'Frame: {frame_idx} | Time: {pose["time"]:.2f}s | Roll: {pose["roll"]:.1f}° Pitch: {pose["pitch"]:.1f}° Yaw: {pose["yaw"]:.1f}°',
            fontsize=12,
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    # --- 4. FuncAnimationの呼び出し ---
    # framesの数は subset_poses の長さに一致させる
    anim = FuncAnimation(
        fig, update, frames=len(subset_poses), interval=interval, repeat=True
    )

    if save_path:
        print(f"{save_path} に保存中...")
        anim.save(save_path, writer="pillow", fps=20)
        print("保存完了")

    return anim


# データ出力用関数
def export_poses_to_csv(poses, output_path="kuwa_poses.csv"):
    """計算した姿勢データをCSVに出力"""
    data = {
        "time": [p["time"] for p in poses],
        "roll": [p["roll"] for p in poses],
        "pitch": [p["pitch"] for p in poses],
        "yaw": [p["yaw"] for p in poses],
        "origin_x": [p["origin"][0] for p in poses],
        "origin_y": [p["origin"][1] for p in poses],
        "origin_z": [p["origin"][2] for p in poses],
    }
    df_out = pd.DataFrame(data)
    df_out.to_csv(output_path, index=False)
    print(f"姿勢データを {output_path} に保存しました")


# デバッグ用：特定フレームの点の位置を確認
def print_frame_info(poses, frame_idx=0):
    """特定フレームの全点の位置を表示"""
    if frame_idx >= len(poses):
        print(f"エラー: フレーム {frame_idx} は存在しません（最大: {len(poses)-1}）")
        return

    pose = poses[frame_idx]
    print(f"\n=== Frame {frame_idx} の点の位置 ===")
    print(f"Point 1: {pose['p1']}")
    print(f"Point 2: {pose['p2']}")
    print(f"Point 3: {pose['p3']}")
    print(f"Point 4: {pose['p4']}")
    print(f"Point 5: {pose['p5']}")
    print(
        f"\n中心位置: {np.mean([pose['p1'], pose['p2'], pose['p3'], pose['p4'], pose['p5']], axis=0)}"
    )

    ###動画を複数保存したいンゴ


# 複数セグメントの動画を保存する関数
def save_segmented_animations(
    poses, time_segments, base_filename="kuwa_segment", interval=50
):
    """
    時間区切り配列に基づき、連続した区間で動画を保存する。

    Parameters:
    poses (list): 全姿勢データ
    time_segments (list): 動画の区切りとなる時刻 (例: [1, 5, 10, 20])
    base_filename (str): 保存ファイル名の基本部分
    interval (int): アニメーションの間隔 (ms)
    """
    if len(time_segments) < 2:
        print("エラー: time_segments には2つ以上の時刻が必要です。")
        return

    # 配列を連続する (開始時刻, 終了時刻) のペアに変換
    segment_pairs = list(zip(time_segments[:-1], time_segments[1:]))

    print(f"--- {len(segment_pairs)} 個の動画セグメントを保存します ---")

    for i, (start_time, end_time) in enumerate(segment_pairs):
        # ファイル名を生成 (例: kuwa_segment_01_0.0s-5.0s.gif)
        output_filename = (
            f"{base_filename}_{i+1:02d}_{start_time:.1f}s-{end_time:.1f}s.gif"
        )
        print(
            f"\n[セグメント {i+1}] {start_time:.1f}s から {end_time:.1f}s までを {output_filename} に保存中..."
        )

        # create_animation 関数を呼び出し、保存パスを指定
        # アニメーションは即座に表示せず、ファイルとして保存される
        anim = create_animation(
            poses,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            save_path=output_filename,  # ここで保存を実行
        )

        # FuncAnimationの save メソッドは、内部でアニメーションを生成し保存する
        # そのため、通常、保存処理が終わると anim オブジェクト自体は破棄されます

    print("\nすべてのセグメントの保存が完了しました。")


# 使用例
if __name__ == "__main__":
    # CSVファイルを読み込む
    csv_path = "2025_1025_dobusarai.csv"  # ← ファイルパスを指定してください
    df = load_data(csv_path)

    # 全フレームの姿勢を計算
    poses = calculate_all_poses(df)
    print(f"計算完了: {len(poses)} フレーム")

    # デバッグ：最初のフレームの情報を表示
    print_frame_info(poses, 0)

    # 特定フレームを可視化（例：最初のフレーム） 最初のフレームの可視化
    # fig1 = visualize_frame(poses[0], frame_idx=0)
    # plt.show()

    # 時系列グラフ
    fig2 = plot_time_series(poses)
    plt.show()

    # 姿勢データをCSVに出力
    export_poses_to_csv(poses, "kuwa_poses.csv")

    start_sec = 5
    end_sec = 20

    # アニメーション（コメントを外して使用）
    print("アニメーション作成中...")
    anim = create_animation(poses, start_time=start_sec, end_time=end_sec, interval=50)
    plt.show()

    # アニメーションをGIFとして保存
    # print("GIF保存中...")
    # anim.save('kuwa_animation.gif', writer='pillow', fps=20)
    # print("保存完了: kuwa_animation.gif")

    ###時間の区切りで動画保存
    segment_times = [5, 10, 15]

    save_segmented_animations(
        poses,
        time_segments=segment_times,
        base_filename="kuwa_segment_output",
        interval=50,  # 20fpsに相当
    )
