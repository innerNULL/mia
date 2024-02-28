# -*- coding: utf-8 -*-
# file: etl_audio_format_converter.py
# date: 2024-02-21
#
# Usage:
# python bin/etl_audio_format_converter.py ./demo_configs/etl_audio_format_converter.json


import pdb
import sys
import os
import json
from typing import Dict, List

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
from audiopipeline.utils import json_objs2jsonl_file


FFMPEG_CMD_TEMP: str = "ffmpeg -i ${IN_PATH} -ar ${SAMPLE_RATE} -ac ${CHANNELS} -sample_fmt s${BIT_DEPTH} ${OUT_PATH}"


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read()) 
    print(configs)

    metadata_path: str = configs["metadata_jsonl_path"]
    output_dir: str = os.path.abspath(configs["output_dir"])
    target_fmt: str = configs["target_fmt"]
    target_sample_rate: int = configs["target_sample_rate"]
    target_bit_depth: int = configs["target_bit_depth"]
    channels: int = configs["channels"]
    ffmpeg: str = configs["ffmpeg"]
    path_col: str = configs["path_col"]
    transcript_col: str = configs["transcript_col"]

    metadata: List[Dict] = [
        json.loads(x) for x in open(metadata_path, "r").read().split("\n")
        if x not in {""}
    ]
    new_metadata: List[Dict] = []

    os.system("mkdir -p %s" % output_dir)

    for record in metadata:
        src_audio_path: str = record[path_col]
        audio_file_name: str = src_audio_path.split("/")[-1].split(".")[0]
        src_audio_fmt: str = src_audio_path.split("/")[-1].split(".")[1]

        out_audio_path: str = os.path.join(
            output_dir, audio_file_name + "." + target_fmt  
        )
        
        if os.path.exists(out_audio_path):
            pass
        else:
            ffmpeg_cmd: str = FFMPEG_CMD_TEMP\
                .replace("${IN_PATH}", src_audio_path)\
                .replace("${SAMPLE_RATE}", str(target_sample_rate))\
                .replace("${CHANNELS}", str(channels))\
                .replace("${BIT_DEPTH}", str(target_bit_depth))\
                .replace("${OUT_PATH}", out_audio_path)
            print("Running: %s" % ffmpeg_cmd)
            os.system(ffmpeg_cmd)
            
        new_record: Dict = {
            path_col: out_audio_path, 
            transcript_col: record[transcript_col]
        }
        new_metadata.append(new_record)
    
    out_metadata_path: str = os.path.join(output_dir, "metadata.jsonl") 
    json_objs2jsonl_file(out_metadata_path, new_metadata)
    print(out_metadata_path)
