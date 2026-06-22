"""
play_amp_motion_physics.py

在 MuJoCo 物理仿真器里，让 G1 强行跟随 npz 的关节轨迹。
预期：很快摔倒，可用于观察 kinodynamic 不匹配现象。

用法:
    mjpython scripts/play_amp_motion_physics.py --npz <file>
    mjpython scripts/play_amp_motion_physics.py --npz <file> --pd-stiffness 200 --pd-damping 8
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer as mj_viewer

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


def load_npz(npz_path):
    data = np.load(npz_path)
    fps_arr = data["fps"]
    fps = float(fps_arr.item()) if fps_arr.size == 1 else float(fps_arr)
    return {
        "fps": fps,
        "joint_pos": data["joint_pos"],
        "body_pos_w": data["body_pos_w"],
        "body_quat_w": data["body_quat_w"],
    }


def play(npz_path, xml_path, loop, mode, pd_kp, pd_kd, scale, gravity_scale, sim_speed, target_zero, duration):
    motion = load_npz(npz_path)
    fps = motion["fps"]
    T = motion["joint_pos"].shape[0]
    dt_npz = 1.0 / fps

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # === 修改重力 ===
    original_grav = model.opt.gravity[2]
    model.opt.gravity[2] = original_grav * gravity_scale
    # === passthrough 模式额外强制 0 重力、0 接触（对照组 C：纯展示 npz 形态）===
    if mode == "passthrough":
        model.opt.gravity[2] = 0.0
        # 关闭所有 contact：把全部 contact 几何的 contype/conaffinity 设为 0
        # 简单做法：调低 contact 求解器
        print(f"[INFO] passthrough 模式：重力=0，仅 FK 渲染，禁用 mj_step")

    print(f"[INFO] npz        = {npz_path}")
    print(f"       XML        = {xml_path}")
    print(f"       fps={fps}, T={T} frames, dt_npz={dt_npz:.4f}s")
    print(f"       mode       = {mode}")
    print(f"       dt         = {model.opt.timestep:.4f}s (substeps={model.opt.iterations})")
    print(f"       pd_kp={pd_kp}, pd_kd={pd_kd}")
    print(f"       gravity    = {model.opt.gravity[2]:.2f} m/s² "
          f"(原始 {original_grav:.2f} × {gravity_scale})")
    print(f"       sim_speed  = {sim_speed}× (1.0=原速, 0.2=慢放5倍)")
    print(f"[INFO] 关闭窗口退出。\n")

    # 找到所有非 freejoint 的关节在 qpos / qvel / qacc 中的地址
    joint_qpos_addr = []
    joint_qvel_addr = []
    joint_actuator_addr = []  # actuator 控制 = qvel
    for jn in G1_29DOF_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            joint_qpos_addr.append(-1)
            joint_qvel_addr.append(-1)
            joint_actuator_addr.append(-1)
        else:
            joint_qpos_addr.append(model.jnt_qposadr[jid])
            joint_qvel_addr.append(model.jnt_dofadr[jid])
            # === 关键修正：actuator 名字是 joint 名字去掉 "_joint" 后缀 ===
            # 例如 joint="left_hip_pitch_joint" → actuator name="left_hip_pitch"
            actuator_name = jn.replace("_joint", "")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            joint_actuator_addr.append(aid)

    # 检查 actuator
    nu = model.nu
    print(f"[INFO] 找到 {nu} 个 actuator")
    if nu == 0:
        print("[WARN] g1.xml 没有 actuator！自动切换到 mj_forward 模式（不是真物理）")
        mode = "kinematic"

    # === 关键检查：actuator 地址是否真的有效 ===
    print("\n[证明 actuator 映射] 前 4 个 actuator 名称和地址：")
    for k in range(min(4, len(joint_actuator_addr))):
        jn = G1_29DOF_JOINT_ORDER[k]
        aid = joint_actuator_addr[k]
        qa = joint_qpos_addr[k]
        if aid >= 0:
            print(f"  [{k}] {jn}: actuator_id={aid}, qpos_addr={qa}, "
                  f"ctrl[actuator_id]={data.ctrl[aid]:.3f}")
        else:
            print(f"  [{k}] {jn}: actuator NOT FOUND")
    print()

    # 把 npz 第 0 帧写入初值
    jp0 = motion["joint_pos"][0]
    for k, addr in enumerate(joint_qpos_addr):
        if addr < 0 or k >= len(jp0):
            continue
        data.qpos[addr] = float(jp0[k])
    data.qpos[0:3] = motion["body_pos_w"][0, 0]
    data.qpos[3:7] = motion["body_quat_w"][0, 0]
    mujoco.mj_forward(model, data)

    # === 关键证明：在 mjData 中保存"目标关节角"，渲染时实时显示 ===
    # 把当前帧的目标值存在一个 numpy 数组里，调试打印和 ghost 渲染用
    target_qpos = data.qpos.copy()  # 当前 qpos（实际）
    ghost_qpos = data.qpos.copy()  # ghost 用的目标 qpos（会逐帧被 npz 覆盖）

    frame = 0
    last_print = time.time()
    fall_height = data.xpos[1, 2]  # 假设 body 1 是 pelvis
    initial_height = data.xpos[1, 2]
    fallen = False
    fall_frame = -1

    with mj_viewer.launch_passive(model, data) as viewer:
        # === 限时：duration 秒后自动退出（<=0 表示不限时）===
        start_time = time.time() if duration > 0 else None
        # === 关键证明：开始时打印 npz 前 5 帧的左膝关节角 ===
        # 证明 npz 里的数据在动
        print("\n[证明 1] npz 自身数据在变化（说明数据流是活的）：")
        for f in [0, 5, 10, 50, 100, 200]:
            if f < T:
                jp_check = motion["joint_pos"][f]
                print(f"  npz frame {f:3d}: L_knee={jp_check[3]:+.3f} L_hip_pitch={jp_check[0]:+.3f} "
                      f"base_pos_x={motion['body_pos_w'][f, 0, 0]:+.3f}")
        print()

        while viewer.is_running():
            # 决定这一帧的目标关节角
            jp = motion["joint_pos"][frame]
            # 保存当前帧的目标关节角（用于打印/可视化）
            target_qpos[7:7+len(jp)] = jp[:]
            # 目标 base 位姿（仅 kinematic 模式用）
            target_base_pos = motion["body_pos_w"][frame, 0]
            target_base_quat = motion["body_quat_w"][frame, 0]

            if mode == "kinematic":
                # === 强行覆盖：所有 qpos 直接被 npz 设定 ===
                for k, addr in enumerate(joint_qpos_addr):
                    if addr < 0 or k >= len(jp):
                        continue
                    data.qpos[addr] = float(jp[k])
                data.qpos[0:3] = target_base_pos
                data.qpos[3:7] = target_base_quat
                # 速度归零，避免累积
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                ghost_qpos[:] = data.qpos[:]  # kinematic 模式下 ghost = 实际

            elif mode == "passthrough":
                # === 对照组 C：纯展示 npz 数据的形态 ===
                # 1. 关闭重力（0.0 仿真）
                # 2. 不调用 mj_step（无积分、无接触力）
                # 3. 只用 mj_forward 算 forward kinematics → 渲染
                # 效果：G1 完美跟随 npz，永不摔倒
                for k, addr in enumerate(joint_qpos_addr):
                    if addr < 0 or k >= len(jp):
                        continue
                    data.qpos[addr] = float(jp[k])
                data.qpos[0:3] = target_base_pos
                data.qpos[3:7] = target_base_quat
                data.qvel[:] = 0.0
                # 只算 forward kinematics，不积分
                mujoco.mj_forward(model, data)
                ghost_qpos[:] = data.qpos[:]
                # === 关键修复：passthrough 模式也要 sleep 受 sim_speed 控制 ===
                # 否则 viewer 跑得比 viewer.sync() 刷新还快
                if sim_speed != 1.0:
                    time.sleep(dt_npz / max(sim_speed, 1e-6))
                ghost_qpos[:] = data.qpos[:]

            elif mode == "pd":
                # === 真物理：只控制关节，base 自由演化 ===
                # === 真物理：只控制关节，base 自由演化 ===
                # 计算 PD 力矩
                for k, addr in enumerate(joint_qpos_addr):
                    if addr < 0 or k >= len(jp) or joint_actuator_addr[k] < 0:
                        continue
                    # === 关键开关：target_zero=True 时目标恒为 0（证明 npz 在驱动）===
                    if target_zero:
                        target = 0.0
                    else:
                        target = float(jp[k])
                    q_err = target - data.qpos[addr]
                    q_vel = data.qvel[joint_qvel_addr[k]]
                    torque = pd_kp * q_err - pd_kd * q_vel
                    data.ctrl[joint_actuator_addr[k]] = torque
                # mj_step 让物理跑一步
                mujoco.mj_step(model, data)

            elif mode == "pd_base":
                # === PD 控制 + base 也被控制（用 strong 弹簧拉回）===
                for k, addr in enumerate(joint_qpos_addr):
                    if addr < 0 or k >= len(jp) or joint_actuator_addr[k] < 0:
                        continue
                    q_err = float(jp[k]) - data.qpos[addr]
                    q_vel = data.qvel[joint_qvel_addr[k]]
                    torque = pd_kp * q_err - pd_kd * q_vel
                    data.ctrl[joint_actuator_addr[k]] = torque
                # base 的平动和旋转不能直接被 actuator 控制（无执行器）
                # 只能靠外力，或者写 qpos（但这又会破坏速度）
                mujoco.mj_step(model, data)

            # 检查摔倒
            current_height = data.xpos[1, 2]
            if not fallen and current_height < initial_height * 0.5:
                fallen = True
                fall_frame = frame

            # === 关键证明：把目标关节角直接显示为一条文字 ===
            # 在 viewer 的左下角显示 "target=[...] actual=[...]"
            # （这是渲染层证明：每个 actuator 都在按 jp[k] 算力矩）
            # sim_speed=1.0 → 跑 dt_npz/model.opt.timestep 步（与 npz 同速）
            # sim_speed=0.2 → 跑 dt_npz*0.2/model.opt.timestep 步（5 倍慢放）
            if mode == "kinematic":
                # kinematic 直接覆盖，不受 sim_speed 影响，只在最后 sleep
                if sim_speed != 1.0:
                    time.sleep(dt_npz / max(sim_speed, 1e-6))
                frame += 1
            else:
                n_substeps_per_npz_frame = (dt_npz * sim_speed) / model.opt.timestep
                for _ in range(max(1, int(round(n_substeps_per_npz_frame)))):
                    mujoco.mj_step(model, data)
                frame += 1

            if frame >= T:
                if loop:
                    frame = 0
                    # 复位
                    jp0 = motion["joint_pos"][0]
                    for k, addr in enumerate(joint_qpos_addr):
                        if addr < 0 or k >= len(jp0):
                            continue
                        data.qpos[addr] = float(jp0[k])
                    data.qpos[0:3] = motion["body_pos_w"][0, 0]
                    data.qpos[3:7] = motion["body_quat_w"][0, 0]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                else:
                    print("[INFO] 播放完毕。")
                    break

            viewer.sync()

            # === 限时检查 ===
            if start_time is not None and time.time() - start_time > duration:
                print(f"\n[INFO] 已到限定时间 {duration}s，自动退出。")
                break
            if time.time() - last_print > 1.0:
                # 计算关节跟踪误差（跟踪程度）
                q_errs = []
                for k, addr in enumerate(joint_qpos_addr):
                    if addr < 0 or k >= len(jp):
                        continue
                    q_errs.append(abs(float(jp[k]) - data.qpos[addr]))
                q_err_mean = float(np.mean(q_errs)) if q_errs else 0.0
                q_err_max = float(np.max(q_errs)) if q_errs else 0.0
                # base 位姿误差
                base_pos_err = np.linalg.norm(target_base_pos - data.xpos[1])
                # === 关键证明：每帧的目标关节角和实际关节角对比 ===
                # 取左膝关节（index 3）作为代表
                lknee_idx = 3
                lk_addr = joint_qpos_addr[lknee_idx]
                lk_target = float(jp[lknee_idx])
                lk_actual = data.qpos[lk_addr] if lk_addr >= 0 else 0.0
                # 取左髋 pitch（index 0）
                lhip_idx = 0
                lh_addr = joint_qpos_addr[lhip_idx]
                lh_target = float(jp[lhip_idx])
                lh_actual = data.qpos[lh_addr] if lh_addr >= 0 else 0.0
                # 检查 ctrl[actuator_id] 是否真的被写入
                lk_act_id = joint_actuator_addr[lknee_idx]
                lh_act_id = joint_actuator_addr[lhip_idx]
                lk_torque = data.ctrl[lk_act_id] if lk_act_id >= 0 else 0.0
                lh_torque = data.ctrl[lh_act_id] if lh_act_id >= 0 else 0.0
                print(f"[f{frame:4d}] h={current_height:.3f}m | "
                      f"joint_err(avg/max)={q_err_mean:.3f}/{q_err_max:.3f}rad | "
                      f"L_knee: target={lk_target:+.2f} actual={lk_actual:+.2f} "
                      f"err={lk_target-lk_actual:+.2f} torque={lk_torque:+7.1f} | "
                      f"L_hip:  target={lh_target:+.2f} actual={lh_actual:+.2f} "
                      f"err={lh_target-lh_actual:+.2f} torque={lh_torque:+7.1f} | "
                      f"{'FALLEN@'+str(fall_frame) if fallen else ''}")
                last_print = time.time()

        viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--xml", type=str,
                        default="src/assets/robots/unitree_g1/xmls/scene_g1.xml",
                        help="必须用含 actuator 的 scene_g1.xml，否则无法真物理仿真")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--mode", type=str, default="pd",
                        choices=["kinematic", "passthrough", "pd", "pd_base"],
                        help="kinematic/passthrough=纯展示 npz; "
                             "pd=真物理+PD控制关节; pd_base=PD+base")
    parser.add_argument("--pd-kp", type=float, default=200.0, help="PD 刚度")
    parser.add_argument("--pd-kd", type=float, default=8.0, help="PD 阻尼")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gravity-scale", type=float, default=1.0,
                        help="重力缩放系数，0.3 = 30%% 月球重力")
    parser.add_argument("--sim-speed", type=float, default=1.0,
                        help="仿真速度倍率，0.2 = 5 倍慢放")
    parser.add_argument("--target-zero", action="store_true",
                        help="PD 目标恒为 0（对照组：证明 npz 数据在驱动）")
    parser.add_argument("--duration", type=float, default=-1.0,
                        help="限定运行时间（秒），<=0 表示无限（手动关 viewer）")
    args = parser.parse_args()

    npz = Path(args.npz)
    xml = Path(args.xml)
    assert npz.exists(), f"找不到 npz: {npz}"
    assert xml.exists(), f"找不到 xml: {xml}"

    play(str(npz), str(xml), args.loop, args.mode, args.pd_kp, args.pd_kd,
         args.scale, args.gravity_scale, args.sim_speed, args.target_zero,
         args.duration)


if __name__ == "__main__":
    main()
