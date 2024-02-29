# -*- coding: utf-8 -*-
# file: hf_audio_dataset_dev.py
# date: 2024-02-27
#
# Usage:
# python dev/audiopipeline/data/hf_audio_dataset_dev.py common_voice_16_1_dev.jsonl


import pdb
import sys
import os
from datasets import set_caching_enabled
from typing import Dict, List
from datasets import DatasetDict, Dataset
from transformers import WhisperProcessor

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../src")
)
from audiopipeline.data import hf_audio_dataset
from audiopipeline.data.hf_audio_dataset import HfAudioDataset


LANG: str = "zh-TW"
TRANSCRIPT_COL: str = "text"
AUDIO_COL: str = "audio"
DURATION_COL: str = "input_length"


if __name__ == "__main__":
    jsonl_data_path: str = sys.argv[1]

    set_caching_enabled(False)

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="mandarin", task="transcribe"
    )

    dataset: HfAudioDataset = HfAudioDataset(
        jsonl_data_path, jsonl_data_path, jsonl_data_path, processor, 
        sampling_rate=16000, 
        lang=LANG, 
        audio_path_col="path", text_col=TRANSCRIPT_COL, 
        audio_col=AUDIO_COL, duration_col=DURATION_COL,
        max_duration=3
    )

    final_data0 = dataset.get_final_datasets([], ["train"])
    final_data1 = dataset.get_final_datasets([], ["train"])

    print(final_data0["train"][0]["text"])
    print(final_data1["train"][0]["text"])
    
    print([x["text"] for x in final_data1["train"]][:50])
    print([x["text"] for x in dataset.get_static_datasets()["train"]][:50])
    pdb.set_trace()
