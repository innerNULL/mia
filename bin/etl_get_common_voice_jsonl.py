# -*- coding: utf-8 -*-
# file: etl_get_common_voice_jsonl.py
# date: 2024-02-05


import sys
import os
import json
import pandas as pd
from typing import Dict, List


AUDIO_URL_TEMP: str = "https://huggingface.co/datasets/mozilla-foundation/common_voice_${VERSION}/resolve/main/audio/${LANG}/${SPLIT}/${LANG}_${SPLIT}_0.tar?download=true"
METADATA_URL_TEMP: str = "https://huggingface.co/datasets/mozilla-foundation/common_voice_${VERSION}/resolve/main/transcript/${LANG}/${SPLIT}.tsv?download=true"


def init(
    workspace_dir: str, hf_token: str, 
    version: str, langs: List[str], splits: List[str]
) -> None:
    os.system("mkdir -p %s" % workspace_dir)

    for lang in langs:
        for split in splits:
            audio_url: str = AUDIO_URL_TEMP\
                .replace("${VERSION}", version)\
                .replace("${LANG}", lang)\
                .replace("${SPLIT}", split)
            metadata_url: str = METADATA_URL_TEMP\
                .replace("${VERSION}", version)\
                .replace("${LANG}", lang)\
                .replace("${SPLIT}", split)
            print("audio_url: %s" % audio_url)
            print("metadata_url: %s" % metadata_url)
            curr_data_dir: str = os.path.join(workspace_dir, version, lang, split)
            
            if os.path.exists(curr_data_dir):
                print("Path '%s' already exists" % curr_data_dir)
                continue
            
            os.system("mkdir -p %s" % curr_data_dir)
            os.system(
                """
                cd {} && wget --header="Authorization: Bearer {}" {} && mv {} {} && tar -xvf {} && mv {} {}
                """.format(
                    curr_data_dir, hf_token, audio_url, 
                    audio_url.split("/")[-1],
                    audio_url.split("/")[-1].split("?")[0],
                    audio_url.split("/")[-1].split("?")[0], 
                    audio_url.split("/")[-1].split("?")[0].split(".")[0], 
                    "audios"
                )
            )
            os.system(
                """
                cd {} && wget --header="Authorization: Bearer {}" {} && mv {} {}
                """.format(
                    curr_data_dir, hf_token, metadata_url, 
                    metadata_url.split("/")[-1], "metadata.tsv"
                )
            )


def build_jsonl_metadata(workspace_dir: str, version: str) -> Dict:
    output: List[Dict] = []
    root_dir: str = os.path.join(workspace_dir, version)
    for lang in os.listdir(root_dir):
        lang_dir: str = os.path.join(root_dir, lang)
        for split in os.listdir(lang_dir):
            split_dir: str = os.path.join(lang_dir, split)
            audio_dir: str = os.path.join(split_dir, "audios")
            raw_metadata_path: str = os.path.join(split_dir, "metadata.tsv")
            raw_metadata: List[Dict] = pd.read_csv(raw_metadata_path, sep="\t")\
                .to_dict(orient="records")
            for record in raw_metadata:
                audio_path: str = os.path.join(audio_dir, record["path"])
                
                if not os.path.exists(audio_path):
                    print("Warning: Audio path '%s' does not exist" % audio_path)
                
                new_metadata: Dict = {}
                new_metadata["path"] = audio_path
                new_metadata["split"] = split
                new_metadata["lang"] = lang
                new_metadata["text"] = record["sentence"]
                output.append(new_metadata)
    return output


if __name__ == "__main__":
    conf: Dict = json.loads(open(sys.argv[1], "r").read())
    
    workspace_dir: str = os.path.abspath(conf["workspace_dir"])
    hf_token: str = conf["hf_token"]
    version: str = conf["version"]
    langs: List[str] = conf["langs"]
    splits: List[str] = conf["splits"]
    output_path: str = conf["output_path"]

    init(workspace_dir, hf_token, version, langs, splits)
    metadata: List[Dict] = build_jsonl_metadata(workspace_dir, version)

    output_file = open(output_path, "w")
    for record in metadata:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_file.close()


    


