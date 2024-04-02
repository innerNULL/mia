# file: train_dev_test_split_for_lines_data.sh
# date: 2024-04-02
#
# A very naive and simple train/dev/test split generator
# with line based datasets.
#
# Line based dataset means each line represent a full sample, 
# JSON line is a good example.
#
# Usage:
# bash ./bin/etl/train_dev_test_splitter_for_lines_data.sh ${LINES_DATA_PATH_LIKE_JSONL} 1000 1000 


set -x


LINES_DATA_PATH=$1
DEV_SPLIT_SIZE=$2
TEST_SPLIT_SIZE=$3


function main() {
  local total_size=$(wc ${LINES_DATA_PATH} | awk '{print $1}')
  local dev_size=${DEV_SPLIT_SIZE}
  local test_size=${TEST_SPLIT_SIZE}
  local train_size=$((total_size - dev_size - test_size))
  local train_split_path=${LINES_DATA_PATH}.train
  local dev_split_path=${LINES_DATA_PATH}.dev
  local test_split_path=${LINES_DATA_PATH}.test
  head -n ${train_size} ${LINES_DATA_PATH} >> ${train_split_path}
  head -n $((total_size - test_size)) ${LINES_DATA_PATH} | tail -n ${dev_size} >> ${dev_split_path} 
  tail -n ${test_size} ${LINES_DATA_PATH} >> ${test_split_path}
}

main
