# -*- coding: utf-8 -*-
# file: eval.py
# date: 2024-03-20


import pdb
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from typing import Dict, List
from torch import Tensor
from transformers import BertModel, BertTokenizer


def load_data(path_or_name: str) -> List[Dict]:
    out: List[Dict] = []
    if os.path.exists(path_or_name):
        if ".csv" in path_or_name:
            out = pd.read_csv(path_or_name).to_dict(orient="records")
        elif ".jsonl" in path_or_name:
            out = [
                json.loads(x) for x in open(path_or_name, "r").read().split("\n")
                if x not in {""}
            ]
        else:
            raise Exception("Not support %s format" % path_or_name)
    else:
        raise Exception("File %s does not exist" % path_or_name)

    return out


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    hf_lm_path_or_name: str = configs["hf_lm_path_or_name"]
    device: str = configs["device"]

    inf_results: List[Dict] = load_data(configs["data_path_or_name"])

    model = BertModel.from_pretrained(hf_lm_path_or_name).to(torch.device(device))
    tokenizer = BertTokenizer.from_pretrained(hf_lm_path_or_name)

    embeddings: List[Dict] = []
    model.eval()
    for record in tqdm(inf_results):
        target_text: str = record[configs["target_text_col"]]
        output_text: str = record[configs["output_text_col"]]
        target_tokens: Tensor = tokenizer.encode_plus(
            target_text, add_special_tokens=True, return_tensors='pt'
        ).to(torch.device(device))
        output_tokens: Tensor = tokenizer.encode_plus(
            output_text, add_special_tokens=True, return_tensors='pt'
        ).to(torch.device(device))

        with torch.no_grad():
            target_embd: Tensor = None
            output_embd: Tensor = None
            if configs["use_cls_embedding"]:
                target_embd = model(**target_tokens)["last_hidden_state"][0, 0, :]
                output_embd = model(**output_tokens)["last_hidden_state"][0, 0, :]
            else:
                target_embd = torch.mean(
                    model(**target_tokens)["last_hidden_state"][:, 1:, :], dim=1
                ).squeeze()
                output_embd = torch.mean(
                    model(**output_tokens)["last_hidden_state"][:, 1:, :], dim=1
                ).squeeze()
            
            cos_sim: float = torch.cosine_similarity(
                target_embd.reshape(1, -1), output_embd.reshape(1, -1)
            ).cpu().tolist()[0]
            #cos_sim: float = 1 - cosine(
            #    target_embd.detach().cpu().numpy(), output_embd.detach().cpu().numpy()
            #)

        embedding: Dict = {
            configs["target_text_col"]: target_text,
            configs["output_text_col"]: output_text,
            "target_embd": target_embd.cpu().tolist(), 
            "output_embd": output_embd.cpu().tolist(),
            "cos_sim": cos_sim
        }
        embeddings.append(embedding)
    
    cos_sims: List[float] = [x["cos_sim"] for x in embeddings]
    print("Avg cos-similarity=%f" % (sum(cos_sims) / len(cos_sims)))




