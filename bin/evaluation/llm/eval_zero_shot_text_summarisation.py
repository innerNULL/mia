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
from typing import Dict, List, Callable
from torchmetrics.text.rouge import ROUGEScore


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


def call_llama_cpp_server_api(url: str, prompt: str, query: str) -> str:
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


def call_ollama_api(url: str, prompt: str, query: str) -> str:
    """
    curl -X POST http://localhost:11434/api/generate -d '{"model": "llama2", "prompt":"Why is the sky blue?"}'
    or 
    curl -X POST http://localhost:11434/api/generate -d '{"model": "llama2", "prompt":"Why is the sky blue?", "stream":false}'
    """
    data: Dict = {
        "model": "llama2",
        "prompt": "%s\n%s" % (prompt, query),
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]


def get_api_caller(backend: str) -> Callable:
    if backend == "llama.cpp":
        return call_llama_cpp_server_api
    elif backend == "ollama":
        return call_ollama_api
    else:
        raise Exception("'%s' is an unsupported backend for now" % backend)


def post_processing(summarisation: str) -> str:
    sections: List[str] = summarisation.split(":")
    if "summary" in sections[0] or "summarisation" in sections[0]:
        sections = sections[1:]
    return ":".join(sections).strip()


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    
    workspace_dir: str = configs["workspace_dir"]
    llm_api: str = configs["llm_api"]
    data_path_or_name: str = configs["data_path_or_name"]
    text_col: str = configs["text_col"]
    summarisation_col: str = configs["summarisation_col"]
    llm_prompts: List[str] = configs["llm_prompts"]
    out_path: str = configs["out_path"]
    backend: str = configs["backend"]
    
    api_caller: Callable = get_api_caller(backend)
    outputs: List[Dict] = []
    if os.path.exists(out_path):
        print("Output file already exists at '%s'" % out_path)
        outputs = [
            json.loads(x) for x in open(out_path, "r").read().split("\n")
            if x not in {""}
        ]
    else:
        data_path: str = data_path_or_name
        if data_path_or_name in DATASETS_URL:
            text_col = DATASETS_META[data_path_or_name]["text_col"]
            summarisation_col = DATASETS_META[data_path_or_name]["summarisation_col"]
            
            os.system("mkdir -p %s" % workspace_dir)
            data_path: str = download_to(workspace_dir, DATASETS_URL[data_path_or_name])
        
        samples: List[Dict] = []
        if data_path.split(".")[-1] == "csv":
            samples = pd.read_csv(data_path).to_dict(orient="records")
        elif data_path.split(".")[-1] == "jsonl":
            samples = [
                json.loads(x) for x in open(data_path, "r").read().split("\n") 
                if x not in {""}
            ]
        else:
            raise Exception("%s is unsupported data formst" % data_path.split(".")[-1])

        for sample in tqdm(samples):
            text: str = sample[text_col]
            summarisation: str = sample[summarisation_col]
            gen_text: str = api_caller(
                llm_api, prompt="", query="\n".join(llm_prompts) + "\n" + text, 
            )
            gen_text = post_processing(gen_text)
            outputs.append(
                {
                    text_col: text, summarisation_col: summarisation, 
                    "gen_text": gen_text
                }
            )

        file = open(out_path, "w")
        for record in outputs:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("See results at '%s'" % out_path)

    metric_obj: ROUGEScore = ROUGEScore()
    metrics: Dict = {
        k: v.cpu().tolist() for k, v in metric_obj(
            [x[text_col] for x in outputs],
            [x[summarisation_col] for x in outputs]
        ).items()
    }
    print(metrics)
