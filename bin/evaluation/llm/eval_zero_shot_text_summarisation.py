# -*- coding: utf-8 -*-
# file: eval_summarisation.py
# date: 2024-03-13


import pdb
import sys
import os
import json
import datasets
import requests
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from typing import Dict, List


DATASETS_URL: Dict[str, str] = {
    "Elfsong/ClinicalDataset train": "https://huggingface.co/datasets/Elfsong/ClinicalDataset/resolve/main/TaskA-TrainingSet.csv",
    "Elfsong/ClinicalDataset dev": "https://huggingface.co/datasets/Elfsong/ClinicalDataset/resolve/main/TaskA-ValidationSet.csv"
}


DATASETS_META: Dict[str, Dict[str, str]] = {
    "Elfsong/ClinicalDataset train": {
        "text_col": "dialogue", "summarisation_col": "section_text"
    },
    "Elfsong/ClinicalDataset dev": {
        "text_col": "dialogue", "summarisation_col": "section_text"
    },
}


def download_to(dir_path: str, url: str) -> str:
    os.system("cd %s && wget %s" % (dir_path, url))
    return os.path.join(dir_path, url.split("/")[-1])


def call_llama_cpp_server_api(
    url: str, prompt: str, query: str
) -> Dict:
    """
    curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json" --data '{"prompt": "hello?", "n_predict": 128}'
    """
    headers: Dict = {"Content-Type": "application/json"}
    data: Dict = {
        "prompt": query,
        "n_predict": 128,
        "system_prompt": {
            "prompt": prompt,
            "anti_prompt": "User:",
            "assistant_name": "Assistant:"
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["content"]


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    
    workspace_dir: str = configs["workspace_dir"]
    llm_api: str = configs["llm_api"]
    data_path_or_name: str = configs["data_path_or_name"]
    text_col: str = configs["text_col"]
    summarisation_col: str = configs["summarisation_col"]
    llm_prompt: str = configs["llm_prompt"]
    out_path: str = configs["out_path"]
    
    data_path: str = data_path_or_name
    if data_path_or_name in DATASETS_URL:
        text_col = DATASETS_META[data_path_or_name]["text_col"]
        summarisation_col = DATASETS_META[data_path_or_name]["summarisation_col"]
        
        os.system("mkdir -p %s" % workspace_dir)
        data_path: str = download_to(workspace_dir, DATASETS_URL[data_path_or_name])

    samples: List[Dict] = pd.read_csv(data_path).to_dict(orient="records")
    outputs: List[Dict] = []
    for sample in tqdm(samples):
        text: str = sample[text_col]
        summarisation: str = sample[summarisation_col]
        gen_text: str = call_llama_cpp_server_api(
            llm_api, prompt=llm_prompt, query=text, 
        )
        outputs.append(
            {
                text_col: text, summarisation_col: summarisation, 
                "gen_text": gen_text
            }
        )

    pdb.set_trace()

