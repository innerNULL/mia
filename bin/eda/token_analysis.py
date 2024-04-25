# -*- coding: utf-8 -*-
# file: token_analysis.py
# date: 2024-04-03
#
# This can help getting some data insights about JSON lines
# dataset's text fields' tokens, the tokenization is based 
# on HuggingFace transformers tokenizer.
#
# Usage:
# python ./bin/eda/token_analysis.py ./demo_configs/eda/token_analysis.json


import pdb
import sys
import os
import json
from tqdm import tqdm
from typing import Dict, List
from pandas import DataFrame
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    data_path: str = configs["jsonl_data_path"]
    target_fields: str = configs["target_fields"]
    max_sample_size: int = configs["max_sample_size"]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs["tokenizer_path_or_name"]
    )

    logs: Dict[ str, List[Dict] ] = {k: [] for k in target_fields}
    file = open(data_path, "r")
    line: str = file.readline()
    cnt: int = 0

    pbar = tqdm(total=max_sample_size)
    while line and cnt < max_sample_size:
        sample: Dict = json.loads(line)
        for field in target_fields:
            text: str = sample[field] 
            tokens: List[int] = tokenizer.encode(
                text, 
                add_special_tokens=False, 
                padding=False, truncation=False, max_length=None, 
                return_tensors=None
            )
            tokens_length: str = len(tokens)
            record: Dict = {
                "tokens_length": tokens_length, 
                "tokens": ",".join([str(x) for x in tokens]),
                "doc_id": cnt
            }
            logs[field].append(record)
        cnt += 1
        pbar.update(1)
        line = file.readline()
    pbar.close()
    file.close()
    
    for field in target_fields:
        print("###################### %s ######################" % field)
        dfrm: DataFrame = DataFrame(logs[field])
        print("Tokens Length Distribution")
        print(dfrm["tokens_length"].describe(
            percentiles=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ))

