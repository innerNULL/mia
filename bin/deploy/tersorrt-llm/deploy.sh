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
    ${CONDA} create --prefix ${PY_RUNTIME} python=3.10 --yes
    rm ${PY_RUNTIME}/compiler_compat/ld
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
  git clone --progress --verbose https://${HF_USER}:${HF_TOKEN}@huggingface.co/${LLM_VERSION}
}


function build_model() {
  cd ${CURR_DIR}
  local model_path=${MODEL_DIR}/$(echo ${LLM_VERSION} | awk -F'/' '{print $NF}')
  local data_type="float16"

  ${PYTHON} ${WORKSPACE}/TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir ${model_path} \
    --output_dir ${model_path}_merged \
    --dtype ${data_type} 

  ${PY_RUNTIME}/bin/trtllm-build \
    --checkpoint_dir ${model_path}_merged \
    --gemm_plugin ${data_type} \
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
