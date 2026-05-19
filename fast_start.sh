#!/usr/bin/env bash
# fast_start.sh — Real2Sim 端到端启动脚本 (red_mug + black_mug_tree 双物体 demo)
#
# 你需要先准备的素材（放到 R2SConfig/assets/ 下，绝对不要写进 MV-SAM3D/sam3）：
#
#   R2SConfig/assets/
#   ├── red_mug/images/          # red mug 多视角照片 (.jpg/.png, 任意名)
#   │   ├── 0.jpg
#   │   ├── 1.jpg
#   │   └── 2.jpg                  (3-5 张就够)
#   ├── black_mug_tree/images/         # mug tree 多视角照片
#   │   ├── 0.jpg
#   │   └── ...                    (3-5 张)
#   └── scene.jpg                # 场景图 (同时包含 red_mug 和 black_mug_tree)
#
# 然后:
#   bash fast_start.sh           # 一口气跑完所有步骤
#   # 或者按 step 拆开:
#   bash fast_start.sh 1         # 只跑 step 1
#   bash fast_start.sh 3c        # 只跑 step 3c
#
# 没 GPU 警告：SAM3 和 SAM3D 大模型推理基本不可能在 CPU 上跑通。
#              脚本会先检查 CUDA，没有就 abort（除非你设 ALLOW_NO_GPU=1）。
export LIDRA_SKIP_INIT=1
set -euo pipefail

# 切到脚本所在目录
cd "$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(pwd)"

# Conda hook (允许在脚本里 conda activate)
if ! command -v conda >/dev/null 2>&1; then
    echo "[fatal] conda 不在 PATH 里"; exit 1
fi
eval "$(conda shell.bash hook)"

# ── 0. Pre-flight: 检查素材就位 ────────────────────────────────────
require_path() {
    if [ ! -e "$1" ]; then
        echo "[fatal] 缺文件/目录: $1"; exit 1
    fi
}
require_path "assets/red_mug/images"
require_path "assets/black_mug_tree/images"
require_path "assets/scene.jpg"
echo "[preflight] assets/ OK"
echo "  red_mug   views: $(ls assets/red_mug/images/   | wc -l)"
echo "  black_mug_tree  views: $(ls assets/black_mug_tree/images/  | wc -l)"

# 检查 conda envs 存在
for ENV in sam3 sam3d-objects; do
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
        echo "[fatal] conda env '$ENV' 不存在"; exit 1
    fi
done
echo "[preflight] conda envs sam3 / sam3d-objects 都存在"

# 检查 GPU
check_gpu() {
    python - <<'PY' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 2)
PY
}

# ── 选择要跑的 step ─────────────────────────────────────────────
STEP="${1:-all}"
echo "[fast_start] target step = $STEP"

# conda activate 会跑 envs/<name>/etc/conda/activate.d/*.sh，里面 binutils
# 等脚本会先引用 ADDR2LINE 之类没设的变量；脚本顶上的 `set -u` 会让它炸掉。
# 包一个 helper：临时关掉 -u 跑 conda activate，跑完再恢复。
safe_conda_activate() {
    set +u
    conda activate "$1"
    set -u
}

run_sam3() {
    safe_conda_activate sam3
    if [ "${ALLOW_NO_GPU:-0}" != "1" ]; then
        if ! check_gpu; then
            echo "[fatal] sam3 env 没检测到 CUDA。要强制 CPU 跑，设 ALLOW_NO_GPU=1 重试 (会非常慢)"; exit 3
        fi
    fi
}

run_p3d() {
    safe_conda_activate sam3d-objects
    if [ "${ALLOW_NO_GPU:-0}" != "1" ]; then
        if ! check_gpu; then
            echo "[fatal] sam3d-objects env 没检测到 CUDA。要强制 CPU 跑，设 ALLOW_NO_GPU=1 重试 (会非常慢)"; exit 3
        fi
    fi
}

# ── Step 1: 物体多视角 → mask  (env: sam3) ─────────────────────
do_step1() {
    echo; echo "########## STEP 1  (sam3 env) ##########"
    run_sam3
    python full_workflow.py --step 1
}

# ── Step 2: 物体 mesh 重建  (env: sam3d-objects) ───────────────
do_step2() {
    echo; echo "########## STEP 2  (sam3d-objects env) ##########"
    run_p3d
    # 让 full_workflow 打印命令以供参考 (写日志)
    python full_workflow.py --step 2

    # 实际执行 SAM3D 重建
    pushd MV-SAM3D > /dev/null
    for OBJ_NAME in red_mug black_mug_tree; do
        INPUT_PATH="$PROJECT_ROOT/workdir/$OBJ_NAME"
        if [ ! -d "$INPUT_PATH/images" ]; then
            echo "[step2] [$OBJ_NAME] skip: $INPUT_PATH/images 不存在 (跑过 step 1 没?)"
            continue
        fi
        # 自动从 $INPUT_PATH 下找 mask 子目录 (排除 images/), step1 按 prompt_slug 命名,
        # 跟 full_workflow.py OBJECTS[*].prompt 保持同步, 不再写死。
        SLUG=$(ls -1 "$INPUT_PATH" | grep -v '^images$' | head -1)
        if [ -z "$SLUG" ] || [ ! -d "$INPUT_PATH/$SLUG" ]; then
            echo "[step2] [$OBJ_NAME] skip: $INPUT_PATH 下找不到 mask 子目录 (step 1 没出 mask?)"
            continue
        fi
        # 从 images 目录推 image_names (按 stem 排序)
        IMAGE_NAMES=$(ls "$INPUT_PATH/images" | sed -E 's/\.[^.]+$//' | sort | paste -sd, -)
        echo "[step2] running SAM3D on $OBJ_NAME (--mask_prompt $SLUG --image_names $IMAGE_NAMES)"
        python run_inference_weighted.py \
            --input_path "$INPUT_PATH" \
            --mask_prompt "$SLUG" \
            --image_names "$IMAGE_NAMES" \
            2>&1 | tee -a "$PROJECT_ROOT/logs/step2_sam3d_${OBJ_NAME}.log"
    done
    popd > /dev/null
}

# ── Step 3a: 场景图 → SAM3 分割  (env: sam3) ───────────────────
do_step3a() {
    echo; echo "########## STEP 3a  (sam3 env) ##########"
    run_sam3
    python full_workflow.py --step 3a
}

# ── Step 3b: 场景图 → metric pointmap (可选, 跳过)  ────────────
do_step3b() {
    echo; echo "########## STEP 3b  (sam3d-objects env, optional) ##########"
    run_p3d
    python full_workflow.py --step 3b
    echo "[step3b] (实际不跑；如需 metric pointmap 改进 Z 轴，按打印命令手动跑)"
}

# ── Step 3d: SAM3D 在 scene 单视角跑一次 → scene 系下 pose + mesh (env: sam3d-objects) ──
do_step3d() {
    echo; echo "########## STEP 3d  (sam3d-objects env) ##########"
    run_p3d
    python step3d_scene_pose.py
}

# ── Step 3e: ICP 把 multi-view mesh 对齐到 scene mesh, 出 init pose (env: sam3d-objects) ──
do_step3e() {
    echo; echo "########## STEP 3e  (sam3d-objects env) ##########"
    run_p3d
    python step3e_align_meshes.py
}

# ── Step 3c: Real2Sim 位姿优化  (env: sam3d-objects) ───────────
do_step3c() {
    echo; echo "########## STEP 3c  (sam3d-objects env) ##########"
    run_p3d
    python full_workflow.py --step 3c
}

# ── Step 4: 渲染对比  (env: sam3d-objects) ──────────────────────
do_step4() {
    echo; echo "########## STEP 4  (sam3d-objects env) ##########"
    run_p3d
    python full_workflow.py --step 4
}

case "$STEP" in
    1)        do_step1 ;;
    2)        do_step2 ;;
    3a)       do_step3a ;;
    3b)       do_step3b ;;
    3c)       do_step3c ;;
    3d)       do_step3d ;;
    3e)       do_step3e ;;
    4)        do_step4 ;;
    all|"")
        do_step1
        do_step3a   # 趁还在 sam3 env，把场景分割一起跑了
        do_step2
        do_step3b
        do_step3d   # SAM3D on scene → scene-frame pose + mesh
        do_step3e   # ICP align multi-view mesh ← scene mesh
        do_step3c
        do_step4
        ;;
    *) echo "用法: bash $0 [1|2|3a|3b|3c|3d|3e|4|all]"; exit 1 ;;
esac

echo
echo "[fast_start] 完成。产物:"
echo "  workdir/                       中间产物 (mask, scene_masks.pt)"
echo "  MV-SAM3D/visualization/        物体 SAM3D mesh"
echo "  outputs/full_workflow/         scene.json + comparison/"
echo "  logs/                          每步日志"
