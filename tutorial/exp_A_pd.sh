#!/bin/bash
# ============================================================================
# 对照实验 A: PD 跟踪 + 真物理仿真
# ============================================================================
# 目的:
#   - 真实物理仿真 (重力/接触/惯性全部开启)
#   - 用 PD 控制把每个关节拉到 npz 的目标角度
#   - base 自由演化 (不控制)
#
# 预期:
#   - G1 关节会跟 npz 动 (力矩不为零)
#   - 但 base 没人控制, 1-2 秒会倒
#   - 用低重力 + 慢放可以看更清楚
#
# 默认运行 30 秒 (足够看完 G1 倒下 + 多次循环)
# 改时长: 在脚本里修改 --duration 后的秒数，或删掉该参数无限运行
# ============================================================================

set -e

# 激活环境 (如果已经在对应环境里可注释掉)
# conda activate mujocolab

# 默认 npz
NPZ="${1:-src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz}"
DURATION="${2:-30}"  # 第二个参数可指定时长 (秒)

# macOS 必须用 mjpython
if [[ "$OSTYPE" == "darwin"* ]]; then
    CMD="mjpython"
else
    CMD="python"
fi

echo "=========================================="
echo " 实验 A: PD + 真物理 (跟随 npz)"
echo " npz: $NPZ"
echo " duration: ${DURATION}s"
echo " cmd: $CMD"
echo "=========================================="

$CMD scripts/play_amp_motion_physics.py \
    --npz "$NPZ" \
    --mode pd \
    --pd-kp 100 \
    --pd-kd 10 \
    --gravity-scale 0.3 \
    --sim-speed 0.2 \
    --loop \
    --duration "$DURATION"
