"""
play_amp_motion.py

最小可运行脚本：直接读取 AMP 专家数据 .npz，在 MuJoCo 中用原生 viewer 播放。
不依赖 mjlab 框架，纯 mujoco + numpy。

使用方法:
    python scripts/play_amp_motion.py --npz src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz
    python scripts/play_amp_motion.py --npz <某个 .npz> --xml src/assets/robots/unitree_g1/xmls/g1_23dof.xml --loop
    python scripts/play_amp_motion.py --npz <某个 .npz> --realtime --scale 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np


# 与 g1.xml (29dof) 中的 joint 顺序一致
# （注意：mjlab 的 npz 数据中 joint_pos 也是按这个顺序保存的）
G1_29DOF_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def load_npz(npz_path: str):
    data = np.load(npz_path)
    # npz 里 fps 是 (1,) 或 0-d 数组，兼容两种
    fps_arr = data["fps"]
    fps = float(fps_arr.item()) if fps_arr.size == 1 else float(fps_arr)
    print(f"[INFO] Loaded {npz_path}")
    print(f"       fps          = {fps:.2f}")
    print(f"       joint_pos    = {data['joint_pos'].shape} (T, 29)")
    print(f"       body_pos_w   = {data['body_pos_w'].shape} (T, B, 3)")
    print(f"       body_quat_w  = {data['body_quat_w'].shape} (T, B, 4)")
    return data, fps


def play(npz_path: str, xml_path: str, loop: bool, realtime: bool, scale: float):
    data, fps = load_npz(npz_path)
    joint_pos = data["joint_pos"]   # (T, num_joints_in_npz)
    body_pos_w = data["body_pos_w"] # (T, B, 3)
    body_quat_w = data["body_quat_w"]  # (T, B, 4) wxyz

    T = joint_pos.shape[0]
    dt = 1.0 / fps

    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_mj = mujoco.MjData(model)

    # 把 mesh 路径打印出来方便确认
    print(f"[INFO] XML         = {xml_path}")
    print(f"       nq={model.nq}, nu={model.nu}, njnt={model.njnt}")
    print(f"       T frames    = {T}, dt = {dt:.4f}s (scale={scale})")
    print(f"[INFO] 按 Ctrl+C 退出 viewer。\n")

    # === 1. 设置 floating base 关节 (freejoint, qpos[0:7]) ===
    # 默认站姿
    base_pos = body_pos_w[0, 0].copy()    # 第 0 个 body 一般是 base/torso
    base_quat = body_quat_w[0, 0].copy()  # wxyz
    data_mj.qpos[0:3] = base_pos
    data_mj.qpos[3:7] = base_quat

    # === 2. 找到每个 joint 在 qpos 中的地址 ===
    joint_qpos_addr = []
    for jn in G1_29DOF_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            print(f"[WARN] joint {jn} not found in XML, skip")
            joint_qpos_addr.append(-1)
        else:
            joint_qpos_addr.append(model.jnt_qposadr[jid])

    npz_num_joints = joint_pos.shape[1]
    print(f"[INFO] XML actuators = {model.nu}, npz joints = {npz_num_joints}")

    # === 3. 进入 viewer 循环 ===
    with mj_viewer.launch_passive(model, data_mj) as viewer:
        frame = 0
        while viewer.is_running():
            # 把当前帧数据写入 qpos
            jp = joint_pos[frame]  # (npz_num_joints,)
            # 直接按顺序写入 joint_qpos_addr 对应的 qpos 位置
            for k, addr in enumerate(joint_qpos_addr):
                if addr < 0 or k >= npz_num_joints:
                    continue
                data_mj.qpos[addr] = float(jp[k])

            # base 状态
            data_mj.qpos[0:3] = body_pos_w[frame, 0]
            data_mj.qpos[3:7] = body_quat_w[frame, 0]

            mujoco.mj_forward(model, data_mj)

            viewer.sync()

            # 帧推进
            if realtime:
                import time
                time.sleep(dt * scale)
            else:
                # 默认 60 fps 渲染（如果想更慢可以加 sleep）
                pass

            frame += 1
            if frame >= T:
                if loop:
                    frame = 0
                else:
                    print("[INFO] 播放完毕，退出。")
                    break

        # viewer 关闭后退出 with 块
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Play AMP expert motion in MuJoCo viewer")
    parser.add_argument("--npz", type=str, required=True, help="AMP .npz 路径")
    parser.add_argument("--xml", type=str,
                        default="src/assets/robots/unitree_g1/xmls/g1.xml",
                        help="G1 MJCF 路径（默认 29dof 的 g1.xml）")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--realtime", action="store_true", help="按原始 fps 实时播放")
    parser.add_argument("--scale", type=float, default=1.0, help="时间缩放，>1 慢放，<1 快放")
    args = parser.parse_args()

    npz = Path(args.npz)
    xml = Path(args.xml)
    assert npz.exists(), f"找不到 npz: {npz}"
    assert xml.exists(), f"找不到 xml: {xml}"

    play(str(npz), str(xml), args.loop, args.realtime, args.scale)


if __name__ == "__main__":
    main()

# === macOS 启动提示 ===
if __name__ == "__main__":
    import sys as _sys, platform
    if platform.system() == "Darwin" and "mjpython" not in _sys.executable:
        print("[HINT] 在 macOS 上需要用 mjpython 启动，例如：")
        print("       mjpython scripts/play_amp_motion.py --npz <file> --realtime --scale 0.3")
        print("       （不要用普通的 python，否则无法打开 viewer）")
