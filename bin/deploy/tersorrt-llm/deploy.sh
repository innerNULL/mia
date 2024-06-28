# file: deploy.sh


set -x


source $1

CURR_DIR=$(pwd)
WORKSPACE="${CURR_DIR}/_workspace"
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
  git clone -b v${TRITON_LLM_VERSION} https://github.com/triton-inference-server/tensorrtllm_backend.git
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
  
  if [ -d "${model_path}_converted" ]; then
    echo "Path ${model_path}_converted already exists"
  else
    ${PYTHON} ${WORKSPACE}/TensorRT-LLM/examples/llama/convert_checkpoint.py \
      --model_dir ${model_path} \
      --output_dir ${model_path}_merged \
      --dtype ${DATA_TYPE}

    ${PY_RUNTIME}/bin/trtllm-build \
      --checkpoint_dir ${model_path}_merged \
      --gemm_plugin ${DATA_TYPE} \
      --output_dir ${model_path}_converted

    rm -rf ${model_path}_merged
  fi
  
  local converted_model_dir=$(cd ${model_path}_converted && pwd)
  cd ${WORKSPACE}/tensorrtllm_backend
  cp ${converted_model_dir}/rank0.engine ./all_models/inflight_batcher_llm/tensorrt_llm/1/
  
  local triton_model_dir="$(pwd)/all_models/inflight_batcher_llm/tensorrt_llm"
  local inf_modes="decoupled_mode:true"
  inf_modes="${inf_modes},engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1"
  inf_modes="${inf_modes},max_tokens_in_paged_kv_cache:"
  inf_modes="${inf_modes},batch_scheduler_policy:guaranteed_completion"
  inf_modes="${inf_modes},kv_cache_free_gpu_mem_fraction:0.2"
  inf_modes="${inf_modes},max_num_sequences:4"
  
  ${PYTHON} tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
    ${inf_modes}
  
  ${PYTHON} tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:${LLM_VERSION}

  ${PYTHON} tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:${LLM_VERSION}
}


function main() {
  init
  download_trtllm_src
  build_py_runtime
  download_model
  build_model
}


main
