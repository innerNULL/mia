# -*- coding: utf-8 -*-
# file: etl_build_med_dialoge_cleaned_text_for_tts.py
# date: 2024-02-22
#
# This program will processing dataset (https://github.com/UCSD-AI4H/Medical-Dialogue-System) 
# to generate a text file for text-to-speech purpose.
# Note to make this easier, we use **preprocess data** mentioned in github repo.


import pdb
import os
import sys
import json
from tqdm import tqdm
from typing import Dict, List

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
from mia.utils import split_text_by_chinese_punctuation


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    raw_file_path: Dict = configs["raw_file_path"]
    out_file_path: Dict = configs["out_file_path"]
    min_chars: int = configs["min_chars"]
    max_chars: int = configs["max_chars"]

    raw_data: List[List[str]] = json.loads(open(raw_file_path, "r").read())
    out_file = open(out_file_path, "w")
    
    for data_group in tqdm(raw_data):
        for record in data_group:
            record = record.replace("病人：", "").replace("医生：", "")
            splitted_texts: List[str] = split_text_by_chinese_punctuation(
                record
            )
            for record in splitted_texts:
                if len(record) < min_chars:
                    continue
                if len(record) > max_chars:
                    record = record[:max_chars]
                out_data: Dict = {"text": record}
                out_file.write(json.dumps(out_data, ensure_ascii=False) + "\n")

    out_file.close()
