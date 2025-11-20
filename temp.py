# 修正後の関数定義
def create_animation(poses, start_frame=0, end_frame=None, interval=50, save_path=None):
    """
    姿勢変化のアニメーションを作成
    start_frame: アニメーションの開始フレーム（インデックス）
    end_frame: アニメーションの終了フレーム（インデックス、このフレームを含む）
    """
    # データをスライスして取得
    if end_frame is None:
        end_frame = len(poses) - 1  # 終点が指定されていなければ最終フレームまで

    # Pythonのスライスは終点を含まないので +1 する
    # また、指定されたフレームが存在するかチェック
    if (
        start_frame < 0
        or start_frame >= len(poses)
        or end_frame >= len(poses)
        or start_frame > end_frame
    ):
        print(
            f"エラー: 無効なフレーム範囲が指定されました (開始: {start_frame}, 終了: {end_frame})"
        )
        return None

    # ここでposesをスライスする
    # end_frameはインデックスなので、Pythonのスライスでは end_frame + 1 を指定する
    subset_poses = poses[start_frame : end_frame + 1]

    if len(subset_poses) == 0:
        print("エラー: 指定された範囲に有効なデータがありません")
        return None

    # ... (以降の処理は subset_poses を使うように変更)

    # 全データの範囲を計算（subset_posesを使用）
    all_points = []
    for pose in subset_poses:  # 変更点: poses -> subset_poses
        all_points.extend([pose["p1"], pose["p2"], pose["p3"], pose["p4"], pose["p5"]])
    # ... (中略) ...

    # FuncAnimationの呼び出し
    # framesの数もsubset_posesの長さに変更
    anim = FuncAnimation(
        fig, update, frames=len(subset_poses), interval=interval, repeat=True
    )

    # ... (以降省略) ...
    return anim
