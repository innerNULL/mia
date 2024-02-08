# -*- coding: utf-8 -*-
# file: crawl_youtube_audio_and_cc_simple.py
# date: 2024-02-07
#
# Usage:
# python ./bin/crawl_youtube_audio_and_cc_simple.py ./demo_configs/crawl_youtube_audio_and_cc_simple.json


import pdb
import sys
import os
import json
import time
import requests
import librosa
from tqdm import tqdm
from typing import Dict, List, Final, Union, Optional

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
from audiopipeline.struct import SubtitleChunk
from audiopipeline.struct import AudioMetadata
from audiopipeline.struct import subtitle_chunk_print
from audiopipeline.struct import subtitle_chunk_new
from audiopipeline.struct import subtitle_chunks_merge
from audiopipeline.struct import audio_metadata_to_json_obj
from audiopipeline.utils import chunk_audio_with_subtitle_chunks


YOUTUBE_DL_BIN: str = os.path.join(
    os.path.dirname(sys.executable), "youtube-dl"
)
YOUTUBE_DL_CMD_TEMP: Final[str] = "${YOUTUBE_DL} --extract-audio --audio-format mp3 --write-sub --all-subs --abort-on-error --output \"${ID}.%(ext)s\" ${URL}"


def get_raw_data(
    raw_data_dir: str, youtube_urls: List[str], lang: str
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    
    for url in tqdm(youtube_urls):
        resource_id: str = url.split("v=")[-1]
        subtitle_path: str = os.path.join(raw_data_dir, "%s.%s.vtt" % (resource_id, lang))
        audio_path: str = os.path.join(raw_data_dir, "%s.mp3" % resource_id)
        
        raw_sample: Dict[str, str] = {
            "id": resource_id, 
            "subtitle_path": subtitle_path, "audio_path": audio_path 
        }
        out.append(raw_sample)

        if os.path.exists(subtitle_path) and os.path.exists(audio_path):
            continue
        
        print("Fetching {}".format(raw_sample))
        cmd: str = YOUTUBE_DL_CMD_TEMP\
            .replace("${YOUTUBE_DL}", YOUTUBE_DL_BIN)\
            .replace("${ID}", resource_id)\
            .replace("${URL}", url)
        os.system("cd %s && %s" % (raw_data_dir, cmd))
    
    out = [
        x for x in out 
        if os.path.exists(x["subtitle_path"]) and os.path.exists(x["audio_path"]) 
    ]
    return out


def chunking_subtitle(path: str) -> List[SubtitleChunk]:
    output: List[Dict] = []
    chunks_metadata: List[str] = [
        x for x in open(path, "r").read().split("\n\n")
        if x not in {""}
    ][1:]
    for chunk_metadata in chunks_metadata:
        time_part: str = ""
        subtitle: str = ""
        start_time: str = ""
        end_time: str = ""

        try:
            time_part, subtitle = chunk_metadata.split("\n")
            start_time, end_time = time_part.split(" --> ")
        except Exception as e:
            print("Warning: %s" % str(e))
            print(chunk_metadata)
            continue

        subtitle_chunk: SubtitleChunk = subtitle_chunk_new(
            start_time, end_time, subtitle
        )
        output.append(subtitle_chunk)
    return subtitle_chunks_merge(output)


if __name__ == "__main__":
    print("Using '%s'" % YOUTUBE_DL_BIN)

    conf: Dict = json.loads(open(sys.argv[1], "r").read())
    print(conf)

    output_dir: str = os.path.abspath(conf["output_dir"])
    lang: str = conf["lang"]
    youtube_urls: List[str] = conf["youtube_urls"]
    
    if not os.path.exists(YOUTUBE_DL_BIN):
        YOUTUBE_DL_BIN = os.path.join(
            os.path.expanduser("~"), ".local", "bin", "youtube-dl"
        )

    raw_data_dir: str = os.path.join(output_dir, "raw")
    dataset_dir: str = os.path.join(output_dir, "dataset")
    os.system("mkdir -p %s" % raw_data_dir)
    os.system("mkdir -p %s" % dataset_dir)
    
    raw_data: List[Dict] = get_raw_data(raw_data_dir, youtube_urls, lang) 

    audios: List[AudioMetadata] = []
    for record in tqdm(raw_data):
        curr_audios: List[AudioMetadata] = chunk_audio_with_subtitle_chunks(
            dataset_dir, 
            record["audio_path"],
            chunking_subtitle(record["subtitle_path"])
        )
        audios += curr_audios
    
    out_metadata_path: str = os.path.join(output_dir, "dataset", "metadata.jsonl")
    out_metadata = open(out_metadata_path, "w")
    for audio in audios:
        json_obj: str = audio_metadata_to_json_obj(audio)
        out_metadata.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    out_metadata.close()

    print("Finished, see metadata at '%s'" % out_metadata_path)
