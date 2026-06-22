#!/bin/bash
# ============================================================================
# 对照实验 C: 纯展示 npz 数据集形态 (passthrough 模式)
# ============================================================================
# 目的:
#   - 禁用所有物理仿真 (重力=0, 无接触, 无积分)
#   - 只读取 npz 的关节角, 通过 mj_forward 算 forward kinematics
#   - 验证 npz 数据本身长什么样, 永远不倒
#
# 预期:
#   - G1 完美按 npz 走路步态播放
#   - 永不倒下
#   - 视觉流畅, 适合"看数据"
#
# 默认运行 60 秒 (循环 6 遍走路序列)
# 改时长: 第二个参数指定秒数
# ============================================================================

set -e

# 激活环境 (如果已经在对应环境里可注释掉)
# conda activate mujocolab

# 默认 npz
NPZ="${1:-src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz}"
DURATION="${2:-60}"  # 第二个参数可指定时长 (秒)

# macOS 必须用 mjpython
if [[ "$OSTYPE" == "darwin"* ]]; then
    CMD="mjpython"
else
    CMD="python"
fi

echo "=========================================="
echo " 实验 C: passthrough (纯展示 npz)"
echo " npz: $NPZ"
echo " duration: ${DURATION}s"
echo " cmd: $CMD"
echo "=========================================="

$CMD scripts/play_amp_motion_physics.py \
    --npz "$NPZ" \
    --mode passthrough \
    --sim-speed 0.3 \
    --loop \
    --duration "$DURATION"
