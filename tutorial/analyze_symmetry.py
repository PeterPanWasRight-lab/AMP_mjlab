"""
analyze_symmetry.py
===================

对比左右腿关节轨迹，自动找最优相位偏移使其对齐，
验证 AMP 专家数据是否符合真实人体步态的左右对称性。

输出:
- /tmp/knee_plots/<npz_name>.png  对比图
- 控制台打印对齐前后的相关性和平均误差

判断标准:
- 最佳 shift 应接近半个步态周期
- 对齐后 corr 应 > 0.85
- 对齐后 mean|diff| 应在 0.05~0.25 rad (3°-14°)

使用方法:
    python tutorial/analyze_symmetry.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# G1 关节顺序（与 npz 的 29 维对应）
G1_29DOF = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
L_KNEE = 3
R_KNEE = 9
L_HIP = 0
R_HIP = 6


def find_best_shift(L, R, max_shift_frac=0.5):
    """找 R 平移多少帧后与 L 最匹配（互相关系数最大）"""
    T = len(L)
    max_shift = int(T * max_shift_frac)
    best_corr = -1
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            l = L[:T - shift]
            r = R[shift:]
        else:
            l = L[-shift:]
            r = R[:T + shift]
        if len(l) < 10:
            continue
        lz = (l - l.mean()) / (l.std() + 1e-8)
        rz = (r - r.mean()) / (r.std() + 1e-8)
        c = float(np.dot(lz, rz) / len(lz))
        if c > best_corr:
            best_corr = c
            best_shift = shift
    return best_shift, best_corr


def analyze(npz_path, save_dir="/tmp/knee_plots"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path)
    jp = d['joint_pos']
    fps = float(d['fps'].item())
    T = jp.shape[0]
    time = np.arange(T) / fps

    name = Path(npz_path).stem

    # 1. 膝关节
    L = jp[:, L_KNEE]
    R = jp[:, R_KNEE]
    shift_knee, corr_knee = find_best_shift(L, R)

    # 2. 髋关节 pitch
    Lh = jp[:, L_HIP]
    Rh = jp[:, R_HIP]
    shift_hip, corr_hip = find_best_shift(Lh, Rh)

    # 3. 应用最优 shift
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{name}  (T={T}  fps={fps:.0f})", fontsize=12)

    # 膝关节原始
    ax = axes[0, 0]
    ax.plot(time, L, label="Left knee",  color="C0", lw=2)
    ax.plot(time, R, label="Right knee", color="C3", lw=2, alpha=0.7)
    ax.set_title(f"Knee - original (mean|diff|={np.abs(L-R).mean():.3f} rad, "
                 f"corr={np.corrcoef(L,R)[0,1]:+.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
    ax.legend(); ax.grid(alpha=0.3)

    # 膝关节对齐
    ax = axes[0, 1]
    if shift_knee > 0:
        L_aligned = L[:T - shift_knee]
        R_shifted = R[shift_knee:]
        t_aligned = time[:T - shift_knee]
    else:
        L_aligned = L[-shift_knee:]
        R_shifted = R[:T + shift_knee]
        t_aligned = time[-shift_knee:]
    ax.plot(t_aligned, L_aligned, label="Left knee",  color="C0", lw=2)
    ax.plot(t_aligned, R_shifted, label=f"Right knee (shifted {shift_knee} frames = {shift_knee/fps*1000:.0f}ms)",
            color="C3", lw=2, alpha=0.7)
    ax.set_title(f"Knee - after best shift (corr={corr_knee:+.3f})\n"
                 f"aligned mean|diff|={np.abs(L_aligned-R_shifted).mean():.3f} rad")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
    ax.legend(); ax.grid(alpha=0.3)

    # 髋关节原始
    ax = axes[1, 0]
    ax.plot(time, Lh, label="Left hip pitch",  color="C0", lw=2)
    ax.plot(time, Rh, label="Right hip pitch", color="C3", lw=2, alpha=0.7)
    ax.set_title(f"Hip pitch - original (mean|diff|={np.abs(Lh-Rh).mean():.3f} rad, "
                 f"corr={np.corrcoef(Lh,Rh)[0,1]:+.3f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
    ax.legend(); ax.grid(alpha=0.3)

    # 髋关节对齐
    ax = axes[1, 1]
    if shift_hip > 0:
        Lh_a = Lh[:T - shift_hip]
        Rh_s = Rh[shift_hip:]
        t_h = time[:T - shift_hip]
    else:
        Lh_a = Lh[-shift_hip:]
        Rh_s = Rh[:T + shift_hip]
        t_h = time[-shift_hip:]
    ax.plot(t_h, Lh_a, label="Left hip pitch",  color="C0", lw=2)
    ax.plot(t_h, Rh_s, label=f"Right hip pitch (shifted {shift_hip} frames = {shift_hip/fps*1000:.0f}ms)",
            color="C3", lw=2, alpha=0.7)
    ax.set_title(f"Hip pitch - after best shift (corr={corr_hip:+.3f})\n"
                 f"aligned mean|diff|={np.abs(Lh_a-Rh_s).mean():.3f} rad")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / f"{name}.png"
    plt.savefig(out, dpi=100)
    print(f"  -> {out}")
    plt.close()

    # 打印结果
    print(f"\n  Knee:  best shift = {shift_knee:4d} frames ({shift_knee/fps*1000:+6.0f}ms), "
          f"corr = {corr_knee:+.3f}, "
          f"aligned mean|diff| = {np.abs(L_aligned-R_shifted).mean():.3f} rad")
    print(f"  Hip:   best shift = {shift_hip:4d} frames ({shift_hip/fps*1000:+6.0f}ms), "
          f"corr = {corr_hip:+.3f}, "
          f"aligned mean|diff| = {np.abs(Lh_a-Rh_s).mean():.3f} rad")

    # 步态周期估计（左膝峰值间距）
    peaks = []
    for i in range(1, T - 1):
        if L[i] > L[i-1] and L[i] > L[i+1] and L[i] > 1.0:
            peaks.append(i)
    if len(peaks) > 1:
        periods = np.diff(peaks)
        avg_period_frames = periods.mean()
        print(f"  Gait period estimate: {avg_period_frames:.1f} frames = {avg_period_frames/fps*1000:.0f}ms")

    return shift_knee, corr_knee, shift_hip, corr_hip


if __name__ == "__main__":
    npzs = [
        "src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz",
        "src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A022.npz",
        "src/assets/motions/g1/amp/WalkandRun/walk_sideway_left_loop_002__A021.npz",
        "src/assets/motions/g1/amp/WalkandRun/arc_walk_left_loop_001__A029.npz",
    ]
    for p in npzs:
        print(f"\n========== {Path(p).name} ==========")
        analyze(p)
