# -*- coding: utf-8 -*-
# file: etl_audio_packaging.py
# date: 2024-03-04
#
# Usage
# python ./bin/etl_audio_packaging.py ./demo_configs/etl_audio_packaging.json


import sys
import os
import json
from tqdm import tqdm
from typing import Dict, List

from audiopipeline.utils import json_objs2jsonl_file


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    
    out_dir: str = configs["output_dir"]
    os.system("mkdir -p %s" % out_dir)

    audio_records: List[Dict] = [
        json.loads(x) for x in open(configs["metadata_path"], "r").read().split("\n")
        if x not in {""}
    ]

    out_metadata_path: str = os.path.join(out_dir, "metadata.jsonl")
    out_metadata: List[Dict]  = []
    for audio_record in tqdm(audio_records):
        path: str = audio_record[configs["audio_path_col"]]
        os.system("cp %s %s" % (path, out_dir))

        audio_file_name: str = path.split("/")[-1]
        out_record: Dict = {configs["audio_path_col"]: audio_file_name}
        for col in configs["other_cols"]:
            out_record[col] = audio_record[col]
        out_metadata.append(out_record)

    json_objs2jsonl_file(out_metadata_path, out_metadata)
    print("Meta-data of packed audios are at %s" % out_metadata_path)

    if configs["compress"]:
        print("Compressing audios")
        os.system("tar -czvf ./%s.tar.gz %s" % (out_dir.split("/")[-1], out_dir))

    print("Can delete packed audios If you don't need it any more")
    print(out_dir)


