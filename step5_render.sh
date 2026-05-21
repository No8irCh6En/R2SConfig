#!/usr/bin/env bash
# step5_render.sh — 用 gsrl-lc env 在 GSRL-exp 目录里跑 render_preview.py,
#                  把结果写进 <run_dir>/genesis_preview/, 日志写 <run_dir>/step5_render.log.
#
# 跟 fast_start.sh 拆开是因为 GSRL-exp 用 src-layout 的 'sim' 包, import 时要把它的
# 根目录 (= GSRL-exp/) 放进 PYTHONPATH; 留在 fast_start.sh 里会污染主 env.
#
# run_dir 来自 pipeline_paths (env: SCENE, PROMPT_TAG, RUN_TAG, RUN).
# 如果调用方 export 了 RUN, 就用它; 否则解析 outputs/runs/latest.
#
# 用法:
#   bash step5_render.sh                           # 默认 config = <run_dir>/gsrl_config.json
#   bash step5_render.sh /path/to/custom.json      # 指定 config; 输出落在该 config 同级 genesis_preview/

# `set -u` 跟 conda env (尤其 gxx_linux-64) 的 deactivate hook 不兼容
# (hook 内部用了 CONDA_BACKUP_CXX 这种没 `+set` guard 的写法). 这里只留 -e -o pipefail.
set -eo pipefail

cd "$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(pwd)"

GSRL_DIR="$PROJECT_ROOT/../GSRL-exp"
RENDER_SCRIPT="$GSRL_DIR/experiments/mug_tree/render_preview.py"

# ── resolve config / output dir / log path ────────────────────────────
if [ $# -ge 1 ]; then
    CONFIG="$1"
    OUT_DIR="$(dirname "$(readlink -f "$CONFIG")")/genesis_preview"
    LOG_FILE="$(dirname "$(readlink -f "$CONFIG")")/step5_render.log"
else
    CONFIG=$(python -m pipeline_paths --field gsrl_config_path)
    OUT_DIR=$(python -m pipeline_paths --field genesis_preview_dir)
    LOG_FILE=$(python -m pipeline_paths --field step5_render_log)
fi

if [ ! -f "$CONFIG" ]; then
    echo "[step5_render] [fatal] config 不存在: $CONFIG"; exit 1
fi
if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "[step5_render] [fatal] render script 不存在: $RENDER_SCRIPT"; exit 1
fi

# Conda hook
if ! command -v conda >/dev/null 2>&1; then
    echo "[step5_render] [fatal] conda 不在 PATH"; exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx gsrl-lc; then
    echo "[step5_render] [fatal] conda env 'gsrl-lc' 不存在"; exit 1
fi

conda activate gsrl-lc

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

# GSRL-exp 用 src-layout, 'sim.tasks.mug_tree' 必须从 GSRL-exp/ 起搜
pushd "$GSRL_DIR" > /dev/null
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# render_preview.py:83 会**无条件**用 --workspace-dir 覆盖 config 里 gs_render.workspace_dir,
# 所以这里从 config 里把它读出来再原样传过去, 保证修改 gsrl_config.json/example/1.json 就够用.
WORKSPACE_DIR=$(python -c "import json,sys; print(json.load(open('$CONFIG'))['env']['gs_render']['workspace_dir'])")

echo "[step5_render] cwd            = $PWD"
echo "[step5_render] PYTHONPATH     = $PYTHONPATH"
echo "[step5_render] config         = $CONFIG"
echo "[step5_render] output-dir     = $OUT_DIR"
echo "[step5_render] workspace-dir  = $WORKSPACE_DIR"
echo "[step5_render] log            = $LOG_FILE"

python "$RENDER_SCRIPT" \
    --config        "$CONFIG" \
    --output-dir    "$OUT_DIR" \
    --workspace-dir "$WORKSPACE_DIR" \
    --no-gs \
    2>&1 | tee "$LOG_FILE"

popd > /dev/null
echo
echo "[step5_render] done. images → $OUT_DIR, log → $LOG_FILE"
