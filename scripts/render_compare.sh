#!/usr/bin/env bash
# render_compare.sh — 用同一个 gsrl_config.json 在 PT3D 和 Genesis 里各渲一张,
# 保存到 <config_dir>/render_compare/{pt3d.png,genesis.png} 供肉眼对比.
#
# 目的: 验证 PT3D 和 Genesis 在 mesh 朝向 / 相机 / scale 完全一致的输入下是否产出一致的图.
#   - 一致 → 渲染管线没问题, mug 朝向偏差来自 step3c 出来的 R/t
#   - 不一致 → step5 compose_pose / Genesis M_LOAD / 还有别的 convention 问题
#
# 用法:
#   bash scripts/render_compare.sh                              # 默认 = outputs/runs/latest/gsrl_config.json
#   bash scripts/render_compare.sh path/to/gsrl_config.json

set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PROJECT_ROOT="$(pwd)"

# ── resolve config ─────────────────────────────────────────────────────
if [ $# -ge 1 ]; then
    CONFIG="$1"
else
    CONFIG=$(python -m pipeline_paths --field gsrl_config_path)
fi
CONFIG="$(readlink -f "$CONFIG")"
if [ ! -f "$CONFIG" ]; then
    echo "[fatal] config 不存在: $CONFIG"; exit 1
fi

OUT_DIR="$(dirname "$CONFIG")/render_compare"
mkdir -p "$OUT_DIR"
echo "[compare] config:   $CONFIG"
echo "[compare] out_dir:  $OUT_DIR"

# ── conda hook ─────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[fatal] conda 不在 PATH"; exit 1
fi
eval "$(conda shell.bash hook)"

safe_activate() {
    set +u
    conda activate "$1"
    set -u
}

# ── 1) PT3D render ─────────────────────────────────────────────────────
echo
echo "########## PT3D render (sam3d-objects) ##########"
safe_activate sam3d-objects
python scripts/render_pt3d_check.py --config "$CONFIG" --output "$OUT_DIR/pt3d.png" \
    2>&1 | tee "$OUT_DIR/pt3d.log"

# ── 2) Genesis render (调用现成 step5_render.sh, 它内部 conda activate gsrl-lc) ─
echo
echo "########## Genesis render (gsrl-lc) ##########"
bash "$PROJECT_ROOT/step5_render.sh" "$CONFIG"

# step5_render.sh 写到 <config_dir>/genesis_preview/<sim_c*.png>; 拷一份到对比目录
GENESIS_SRC="$(dirname "$CONFIG")/genesis_preview"
if compgen -G "$GENESIS_SRC/sim_*.png" > /dev/null; then
    cp "$GENESIS_SRC"/sim_*.png "$OUT_DIR/genesis.png"
fi

echo
echo "[compare] DONE."
echo "  PT3D:    $OUT_DIR/pt3d.png"
echo "  Genesis: $OUT_DIR/genesis.png"
echo
echo "用 viewer 打开两张图肉眼对比. 期望: 同样朝向 / 同样位置, 只有 lighting / 材质不同."
