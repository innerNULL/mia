# -*- coding: utf-8 -*-
# file: hf_audio_dataset_dev.py
# date: 2024-02-27
#
# Usage:
# python dev/audiopipeline/data/hf_audio_dataset_dev.py common_voice_16_1_dev.jsonl


import pdb
import sys
import os
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

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="mandarin", task="transcribe"
    )

    """
    raw_datasets: DatasetDict = hf_audio_dataset.datasetdict_load_jsonl(
        jsonl_data_path, jsonl_data_path, jsonl_data_path
    )
    print("Finished generate `raw_datasets`")
    
    for split in raw_datasets:
        dataset: Dataset = raw_datasets[split]
        print(dataset[0])
        dataset = hf_audio_dataset.dataset_load_audio(dataset)
        print(dataset[0])
        dataset = hf_audio_dataset.dataset_audio_time_domain_argumentation(dataset)
        print(dataset[0])
        dataset = hf_audio_dataset.dataset_raw_transcript_processor(
            dataset, text_col=TRANSCRIPT_COL, lang=LANG
        )
        print(dataset[0])
        dataset = hf_audio_dataset.dataset_run_hf_processor(
            dataset, processor, AUDIO_COL, TRANSCRIPT_COL, DURATION_COL
        )
        #print(dataset[0])
    """
    dataset: HfAudioDataset = HfAudioDataset(
        jsonl_data_path, jsonl_data_path, jsonl_data_path, processor, 
        sampling_rate=16000, 
        lang=LANG, 
        audio_path_col="path", text_col=TRANSCRIPT_COL, 
        audio_col=AUDIO_COL, duration_col=DURATION_COL,
        max_duration=3
    )
    final_datasets: DatasetDict = dataset.get_final_datasets()
    train_dataset: Dataset = final_datasets["train"]
    print(train_dataset[0]["text"])
    print(dataset.get_final_datasets()["train"][0]["text"])
    #pdb.set_trace()
