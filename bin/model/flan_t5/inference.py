# -*- coding: utf-8 -*-
# file: inference.py
# date: 2024-04-03
#
# Originally submitted on git@github.com:innerNULL/mia.git
#
# Usage:
# python bin/model/flan_t5/inference.py ./demo_configs/model/flan_t5/inference.json


import pdb
import sys
import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


def dataset_load(path_or_name: str, split: Optional[str]=None) -> List[Dict]:
    output: List[Dict] = []
    if os.path.exists(path_or_name):
        if path_or_name.split(".")[-1] == "csv":
            output = pd.read_csv(path_or_name).to_dict(orient="records")
        elif path_or_name.split(".")[-1] == "jsonl":
            output = [
                json.loads(x) for x in open(path_or_name, "r").read().split("\n")
                if x not in {""}
            ]
        else:
            raise Exception("Not a supported file format")
    else:
        if split is None or split == "":
            raise "Can not loading HuggingFace dataset without split info"
        output = [x for x in load_dataset(path_or_name, split=split)]
    return output


def model_inference_with_decoding(
    model: PreTrainedModel, tokenizer: T5Tokenizer, input_text: str
) -> str:
    model_inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**model_inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    
    dataset: List[Dict] = dataset_load(
        configs["data_path_or_name"], configs["data_split"]
    )
    model: PreTrainedModel = T5ForConditionalGeneration.from_pretrained(
        configs["model_path_or_name"]
    )
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        configs["tokenizer_path_or_name"]
    )
    
    output_records: List[Dict] = []
    for sample in tqdm(dataset):
        input_text: str = "\n".join(
            [configs["prompt"], sample[configs["input_text_col"]]]
        )
        target_text: str = sample[configs["target_text_col"]]
        output_text: str = model_inference_with_decoding(
            model, tokenizer, input_text
        )
        output_record: Dict = {
            configs["input_text_col"]: input_text, 
            configs["target_text_col"]: target_text, 
            "output_text": output_text
        }
        output_records.append(output_record)

    out_file = open(configs["output_path"], "w")
    for record in output_records:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_file.close()
