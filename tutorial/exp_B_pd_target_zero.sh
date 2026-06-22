#!/bin/bash
# ============================================================================
# 对照实验 B: PD 跟踪 + 真物理 + 目标恒为 0 (不接 npz)
# ============================================================================
# 目的:
#   - 与实验 A 完全相同设置
#   - 唯一区别: PD 目标被强制设为 0, npz 数据完全不进入控制回路
#   - 这是"证明 npz 真的被使用"的对照实验
#
# 预期:
#   - G1 关节不跟 npz 动 (保持 zero pose)
#   - 与实验 A 的"挥手动脚"形成鲜明对比
#
# 默认运行 30 秒
# ============================================================================

set -e

NPZ="${1:-src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz}"
DURATION="${2:-30}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    CMD="mjpython"
else
    CMD="python"
fi

echo "=========================================="
echo " 实验 B: PD + 真物理 + 目标=0 (对照组)"
echo " npz: $NPZ (虽然指定了但不使用)"
echo " duration: ${DURATION}s"
echo "=========================================="

$CMD scripts/play_amp_motion_physics.py \
    --npz "$NPZ" \
    --mode pd \
    --pd-kp 100 \
    --pd-kd 10 \
    --gravity-scale 0.3 \
    --sim-speed 0.2 \
    --target-zero \
    --duration "$DURATION"
