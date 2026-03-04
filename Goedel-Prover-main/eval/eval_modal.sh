INPUT_PATH=datasets/minif2f.jsonl
MODEL_PATH=Goedel-LM/Goedel-Prover-SFT
OUTPUT_DIR=results/minif2f/Godel-Prover-SFT-modal
SPLIT=test
N=32
CPU=128 #32
GPU=1
FIELD=complete
MODAL_URL=${MODAL_URL:-}
MODAL_TOKEN_ENV=${MODAL_TOKEN_ENV:-MODAL_API_TOKEN}
MAX_BATCH_SIZE=16
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=2048

while getopts ":i:m:o:s:n:c:g:u:t:b:T:p:k:" opt; do
  case $opt in
    i) INPUT_PATH="$OPTARG"
    ;;
    m) MODEL_PATH="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    s) SPLIT="$OPTARG"
    ;;
    n) N="$OPTARG"
    ;;
    c) CPU="$OPTARG"
    ;;
    g) GPU="$OPTARG"
    ;;
    u) MODAL_URL="$OPTARG"
    ;;
    t) MODAL_TOKEN_ENV="$OPTARG"
    ;;
    b) MAX_BATCH_SIZE="$OPTARG"
    ;;
    T) TEMPERATURE="$OPTARG"
    ;;
    p) TOP_P="$OPTARG"
    ;;
    k) MAX_TOKENS="$OPTARG"
    ;;
  esac
done

if [ -z "$MODAL_URL" ]; then
  echo "MODAL_URL is required. Set env MODAL_URL or pass -u."
  exit 1
fi

python -m eval.step1_inference \
  --input_path "${INPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --n "${N}" \
  --gpu "${GPU}" \
  --provider modal \
  --modal_url "${MODAL_URL}" \
  --modal_token_env "${MODAL_TOKEN_ENV}" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}"

INPUT_FILE=${OUTPUT_DIR}/to_inference_codes.json
COMPILE_OUTPUT_PATH=${OUTPUT_DIR}/code_compilation.json
python -m eval.step2_compile --input_path "$INPUT_FILE" --output_path "$COMPILE_OUTPUT_PATH" --cpu "$CPU"

SUMMARIZE_OUTPUT_PATH=${OUTPUT_DIR}/compilation_summarize.json
python -m eval.step3_summarize_compile --input_path "$COMPILE_OUTPUT_PATH" --output_path "$SUMMARIZE_OUTPUT_PATH" --field "${FIELD}"
