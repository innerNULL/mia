# file: deploy.sh


set -x


source $1

CURR_DIR=$(pwd)
WORKSPACE="./_workspace"
PY_RUNTIME="${WORKSPACE}/pyenv"
PYTHON=${PY_RUNTIME}/bin/python
HF_CLI=${PY_RUNTIME}/bin/huggingface-cli
MODEL_DIR=${WORKSPACE}/models
TRTLLM_SRC_DIR=${WORKSPACE}/TensorRT-LLM


function init() {
  mkdir -p ${WORKSPACE}
  mkdir -p ${MODEL_DIR}
}


function download_trtllm_src() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  git clone -b v${TENSORRT_LLM_VERSION} https://github.com/NVIDIA/TensorRT-LLM.git
}


function build_py_runtime() {
  cd ${CURR_DIR}
  if [ -d "${PY_RUNTIME}" ]; then
    echo "Python env '${PY_RUNTIME}' is already there"
  else
    python3 -m venv ${PY_RUNTIME} --copies
    ${PYTHON} -m pip install --upgrade pip
    ${PYTHON} -m pip install tensorrt_llm==${TENSORRT_LLM_VERSION} -U --extra-index-url https://pypi.nvidia.com
  fi
}


function download_model() {
  cd ${CURR_DIR}
  git lfs install
  ${HF_CLI} login --token ${HF_TOKEN}
  cd ${CURR_DIR}
  cd ${MODEL_DIR}
  git clone https://${HF_USER}:${HF_TOKEN}@huggingface.co/${LLM_VERSION}
}


function build_model() {
  cd ${CURR_DIR}
  local model_path=${MODEL_DIR}/$(echo ${LLM_VERSION} | awk -F'/' '{print $NF}')
  ${PYTHON} ${WORKSPACE}/TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir ${model_path} \
    --output_dir ${model_path}_merged \
    --dtype bfloat16 \
    --load_model_on_cpu

  ${PY_RUNTIME}/bin/trtllm-build \
    --checkpoint_dir ${model_path}_merged \
    --gemm_plugin bfloat16 \
    --output_dir ${model_path}_converted
}


function main() {
  init
  download_trtllm_src
  build_py_runtime
  download_model
  build_model
}


main
