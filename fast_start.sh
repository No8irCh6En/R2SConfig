#!/usr/bin/env bash
# fast_start.sh — Real2Sim 端到端启动脚本.
#
# 你需要先准备的素材:
#
#   assets/
#   ├── scenes/<scene_id>/scene.jpg          # 场景图
#   └── objects/<obj_name>/images/*.{jpg,png}  # 每个物体的多视角照片
#
# 默认 SCENE = assets/scenes/ 下唯一子目录 (多个时报错, 用 SCENE=<scene_id> 指定).
# OBJECTS / SCENE_PROMPT 等在 pipeline_config.py.
#
# 环境变量 (env > 脚本默认):
#   SCENE        scene_id; 默认 = 单 assets/scenes/* 子目录
#   PROMPT_TAG   prompt 子目录前缀; 默认 "auto"
#   RUN_TAG      run 目录后缀; 默认 ""
#   MESHES       "name=path ..." 覆盖某些 object 的 mesh (透传 --meshes/--mesh)
#   FREEZE_SCALE_<name>  1 → 冻结该 object 的 scale, 只优化 R/t
#   INIT_SCALE_<name>    覆盖该 object 的 init_scale (配合 FREEZE_SCALE 用)
#   YAW_ONLY             1 → 所有 object: R 退化为 yaw-only (绕 gravity), roll/pitch=0
#   YAW_ONLY_<name>      1 → 只对该 object 启用 yaw_only
#   YAW_USE_PCA          1 → yaw_only 时用 PCA 长轴当 up (默认信任 mesh canonical +Y)
#   FLOOR_LOCK           1 → t 锁在 plane_z 平面 (mug 底始终贴桌), 只优化 (x, y) 2DOF
#   FLOOR_LOCK_<name>    1 → 只对该 object 启用 floor_lock (需配合 YAW_ONLY=1)
#   SKIP_M_LOAD          1 → PT3D + Genesis 都不应用 M_LOAD, mesh 保持 .glb 原始朝向
#                          (用来测 "如果两边都不动 mesh 朝向, 杯子在 Genesis 里是什么样")
#   WORLD_FRAME          1 → 优化器整个在 Genesis world 里跑, 相机摆 single_pos, R/t 直接是
#                          mesh→Genesis-world; step5 不走 compose_pose. 推荐配合 YAW_ONLY=1
#                          FLOOR_LOCK=1 一起用 (= 4 DOF: θ + t_x + t_y + s)
#   GRAVITY_SRC          auto(默认)/cam/ransac: gravity 怎么估
#                          auto = 先 cam, 失败 fallback ransac (推荐)
#                          cam = 从 example/1.json 相机外参反推, 失败直接报错
#                          ransac = 老 MoGe pointmap RANSAC 拟合最大平面
#   K_SRC                auto(默认)/cam/moge: 相机内参 K 怎么算
#                          auto = 先 cam (从 single_fov 反推), 失败 fallback moge
#                          cam = 强制从 Genesis single_fov, 失败报错
#                          moge = 老 MoGe 估的 (从图像内容反推, 易偏)
#   REBUILD      1 → 忽略 build 缓存, 全部重跑
#   ALLOW_NO_GPU 1 → 没 GPU 也强跑 (会非常慢)
#
# 用法:
#   bash fast_start.sh           # 一口气跑完所有步骤
#   bash fast_start.sh 1         # 只跑 step 1
#   bash fast_start.sh 3c        # 只跑 step 3c (写 build_prompt_dir/scene.json)
#   bash fast_start.sh 5         # 只跑 step 5 (从 build_prompt_dir 读 scene.json,
#                                # 在新 run_dir 出 gsrl_config + genesis_preview)
export LIDRA_SKIP_INIT=1
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(pwd)"

# Conda hook (允许在脚本里 conda activate)
if ! command -v conda >/dev/null 2>&1; then
    echo "[fatal] conda 不在 PATH 里"; exit 1
fi
eval "$(conda shell.bash hook)"

# ── resolve scene_id + 把 SCENE/PROMPT_TAG/RUN_TAG export 给 child python ──
# (real2sim/io/scenes.py 也读 SCENE 来挑 per-scene OBJECTS, 必须 export 才能传下去)
SCENE_ID=$(SCENE="${SCENE:-}" python -m real2sim.io.paths --field scene_id)
export SCENE="$SCENE_ID"
export PROMPT_TAG="${PROMPT_TAG:-auto}"
export RUN_TAG="${RUN_TAG:-}"
PROMPT_ID=$(python -m real2sim.io.paths --field prompt_id)
echo "[preflight] SCENE      = $SCENE  (exported)"
echo "[preflight] PROMPT_TAG = $PROMPT_TAG  (exported)"
echo "[preflight] RUN_TAG    = $RUN_TAG  (exported)"
echo "[preflight] prompt_id  = $PROMPT_ID"
echo "[preflight] OBJECTS    = $(python -c "from real2sim.io.scenes import OBJECTS; print(OBJECTS)")"

# ── 0. Pre-flight: 检查素材就位 ────────────────────────────────────
require_path() {
    if [ ! -e "$1" ]; then
        echo "[fatal] 缺文件/目录: $1"; exit 1
    fi
}

SCENE_IMG=$(python -m real2sim.io.paths --field scene_image)
require_path "$SCENE_IMG"

# 每个 OBJECTS 的 images 目录都要在
while IFS= read -r OBJ_NAME; do
    IMG_DIR=$(python -m real2sim.io.paths --field object_images_dir --obj "$OBJ_NAME")
    require_path "$IMG_DIR"
    N=$(find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
    echo "[preflight]   object $OBJ_NAME views: $N  ($IMG_DIR)"
done < <(python -c "from real2sim.io.scenes import OBJECTS; [print(o['name']) for o in OBJECTS]")

echo "[preflight] scene.jpg = $SCENE_IMG"

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

# ── 分配 run_dir (仅当 step 牵涉 3c/4/5/all) ──────────────────────────
# step3c 把核心数据 (scene.json / pipeline_results.json) 写到 build_prompt_dir
# (跨 run 共享, deterministic), 但 per-object QA 可视化 (per_object/<name>_eval.png)
# 是 per-run viz, 要写到 run_dir, 所以 step3c 也需要 run_dir.
# 历史: 之前 cp -al seed scene.json 到 run_dir, 但 cp -al 是 hardlink, Python
# write_text 走 O_TRUNC 直接 truncate 同一个 inode, 导致 run A 的 step3c 把
# run B 的 scene.json 一起改了. 现在 scene.json 走 build/, 不再 seed, 不会 clobber.
case "$STEP" in
    3c|4|5|all|"") NEEDS_RUN_DIR=1 ;;
    *)             NEEDS_RUN_DIR=0 ;;
esac

if [ "$NEEDS_RUN_DIR" -eq 1 ]; then
    NEW_RUN_DIR=$(python -m real2sim.io.paths --new-run-dir)
    export RUN="$NEW_RUN_DIR"
    echo "[fast_start] RUN  = $RUN"
else
    echo "[fast_start] (step $STEP 不需要 run_dir; outputs/runs/ 不变)"
fi

# ── conda activate 兼容 hook 的 -u 问题 ───────────────────────────────
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
            echo "[fatal] sam3 env 没检测到 CUDA。要强制 CPU 跑，设 ALLOW_NO_GPU=1 重试"; exit 3
        fi
    fi
}

run_p3d() {
    safe_conda_activate sam3d-objects
    if [ "${ALLOW_NO_GPU:-0}" != "1" ]; then
        if ! check_gpu; then
            echo "[fatal] sam3d-objects env 没检测到 CUDA。要强制 CPU 跑，设 ALLOW_NO_GPU=1 重试"; exit 3
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
    # 打印命令供参考 (写日志)
    python full_workflow.py --step 2

    # 实际执行 SAM3D 重建. 每个 OBJECTS 的 build_object_dir 当 --input_path.
    pushd MV-SAM3D > /dev/null
    while IFS=$'\t' read -r OBJ_NAME INPUT_PATH SLUG; do
        if [ ! -d "$INPUT_PATH/images" ]; then
            echo "[step2] [$OBJ_NAME] skip: $INPUT_PATH/images 不存在 (跑过 step 1 没?)"
            continue
        fi
        if [ ! -d "$INPUT_PATH/$SLUG" ]; then
            echo "[step2] [$OBJ_NAME] skip: $INPUT_PATH/$SLUG mask 目录不存在 (step 1 没出 mask?)"
            continue
        fi
        IMAGE_NAMES=$(ls "$INPUT_PATH/images" | sed -E 's/\.[^.]+$//' | sort | paste -sd, -)
        echo "[step2] running SAM3D on $OBJ_NAME (--mask_prompt $SLUG --image_names $IMAGE_NAMES)"
        python run_inference_weighted.py \
            --input_path "$INPUT_PATH" \
            --mask_prompt "$SLUG" \
            --image_names "$IMAGE_NAMES" \
            2>&1 | tee -a "$PROJECT_ROOT/logs/step2_sam3d_${OBJ_NAME}.log"
    done < <(PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" python -c "
from real2sim.io.scenes import OBJECTS
from real2sim.io.paths import resolve, sanitize_filename
paths = resolve()
for o in OBJECTS:
    print(f\"{o['name']}\t{paths.build_object_dir(o['name'], o['prompt']).resolve()}\t{sanitize_filename(o['prompt'])}\")
")
    popd > /dev/null

    # 把每个物体 build 目录下的 mesh.glb symlink 指向最新 result.glb,
    # 让 step3c/3e/5 都能稳定找到.
    python -c "from full_workflow import link_object_mesh, resolve_paths; link_object_mesh(resolve_paths())"
}

# ── Step 3a: 场景图 → SAM3 分割  (env: sam3) ───────────────────
do_step3a() {
    echo; echo "########## STEP 3a  (sam3 env) ##########"
    run_sam3
    python full_workflow.py --step 3a
}

# ── Step 3b: 场景图 → metric pointmap (MoGe)  ─────────────────
do_step3b() {
    echo; echo "########## STEP 3b  (sam3d-objects env, optional) ##########"
    run_p3d
    python full_workflow.py --step 3b
}

# ── Step 3d: SAM3D 在 scene 单视角跑一次 → scene 系下 pose + mesh ──
do_step3d() {
    echo; echo "########## STEP 3d  (sam3d-objects env) ##########"
    run_p3d
    python -m real2sim.pose.scene_pose
}

# ── Step 3e: ICP 把 multi-view mesh 对齐到 scene mesh, 出 init pose ──
do_step3e() {
    echo; echo "########## STEP 3e  (sam3d-objects env) ##########"
    run_p3d
    python -m real2sim.pose.align_meshes
}

# ── Step 3c: Real2Sim 位姿优化  (env: sam3d-objects) ───────────
# MESHES env var: 空格分隔的 "name=path" 对, 会透传给 --meshes (override 模式).
# 例: MESHES="red_mug=example/mug_1.obj" bash fast_start.sh 3c
do_step3c() {
    echo; echo "########## STEP 3c  (sam3d-objects env) ##########"
    run_p3d
    local mesh_args=()
    if [ -n "${MESHES:-}" ]; then
        for pair in $MESHES; do mesh_args+=(--meshes "$pair"); done
    fi
    python full_workflow.py --step 3c "${mesh_args[@]+"${mesh_args[@]}"}"
}

# ── Step 4: 渲染对比  (env: sam3d-objects) ──────────────────────
do_step4() {
    echo; echo "########## STEP 4  (sam3d-objects env) ##########"
    run_p3d
    local mesh_args=()
    if [ -n "${MESHES:-}" ]; then
        for pair in $MESHES; do mesh_args+=(--meshes "$pair"); done
    fi
    python full_workflow.py --step 4 "${mesh_args[@]+"${mesh_args[@]}"}"
}

# ── Step 5: 出 Genesis config + 用 gsrl-lc env 渲染 preview ──────
do_step5() {
    echo; echo "########## STEP 5  (sam3d-objects + gsrl-lc envs) ##########"

    # 5a: 生成 gsrl_config.json (写到 RUN/gsrl_config.json)
    run_p3d
    local mesh_args=()
    if [ -n "${MESHES:-}" ]; then
        for pair in $MESHES; do mesh_args+=(--mesh "$pair"); done
    fi
    python -m real2sim.export.genesis_config "${mesh_args[@]+"${mesh_args[@]}"}" 2>&1 | tee "$PROJECT_ROOT/logs/step5_config.log"

    # 5a.5: 趁还在 sam3d-objects env, 用 PT3D 也渲一张 (跟 Genesis 同相机, 方便对比)
    local gsrl_config
    gsrl_config=$(python -m real2sim.io.paths --field gsrl_config_path)
    local pt3d_out
    pt3d_out="$(dirname "$gsrl_config")/pt3d_preview.png"
    python scripts/render_pt3d_check.py --config "$gsrl_config" --output "$pt3d_out" \
        2>&1 | tee "$PROJECT_ROOT/logs/step5_pt3d_preview.log" || \
        echo "[step5] PT3D preview 失败, 不影响 Genesis 渲染. Genesis 这边继续."

    # 5b: Genesis render (独立脚本, 自己处理 conda activate gsrl-lc / PYTHONPATH)
    bash "$PROJECT_ROOT/step5_render.sh"

    echo
    echo "[step5] 两份对比图:"
    echo "  PT3D:    $pt3d_out"
    echo "  Genesis: $(dirname "$gsrl_config")/genesis_preview/sim_c0.png"
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
    5)        do_step5 ;;
    all|"")
        do_step1
        do_step3a   # 趁还在 sam3 env，把场景分割一起跑了
        do_step2
        do_step3b
        do_step3d   # SAM3D on scene → scene-frame pose + mesh
        do_step3e   # ICP align multi-view mesh ← scene mesh
        do_step3c
        do_step4
        do_step5
        ;;
    *) echo "用法: bash $0 [1|2|3a|3b|3c|3d|3e|4|5|all]"; exit 1 ;;
esac

# ── 成功收尾: 更新 outputs/runs/latest symlink ─────────────────────────
if [ "$NEEDS_RUN_DIR" -eq 1 ] && [ -d "${RUN:-}" ]; then
    python -m real2sim.io.paths --update-latest "$RUN" > /dev/null
    echo
    echo "[fast_start] outputs/runs/latest → $RUN"
fi

echo
echo "[fast_start] 完成. 产物布局:"
echo "  outputs/build/scenes/$SCENE_ID/moge/                 MoGe (跨 prompt 复用)"
echo "  outputs/build/scenes/$SCENE_ID/prompts/$PROMPT_ID/   prompt-相关 + step3c 输出"
echo "                                                       (scene.json / pipeline_results.json)"
echo "  outputs/build/objects/<obj>__<hash>/                 SAM3/SAM3D (跨 scene 复用)"
if [ "$NEEDS_RUN_DIR" -eq 1 ]; then
    echo "  $RUN                                                  本次 run 产物 (gsrl_config / preview / comparison)"
    echo "  outputs/runs/latest -> $(basename "$RUN")"
fi
echo "  logs/                                                 各 step 日志"
