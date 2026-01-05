
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config_stage2.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

echo "[Stage2] Using config: ${CONFIG_PATH}"
python -u "${SCRIPT_DIR}/train.py" --config "${CONFIG_PATH}"
