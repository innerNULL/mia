# -*- coding: utf-8 -*-
# file: test_functions.py
# date: 2024-03-06


import pdb
import librosa
from typing import Dict, List
from torch import Tensor
from opencc import OpenCC
from datasets import DatasetDict, Dataset
from transformers import WhisperProcessor

from audiopipeline.data import functions as F
from audiopipeline.utils import jsonl_file2json_objs


DEMO_JSONL_PATH: str = "./demo_data/demo_jsonl_dataset.jsonl"
DEMO_JSONL_DATA: List[Dict] = jsonl_file2json_objs(DEMO_JSONL_PATH)
AUDIO_PATH_COL: str = "path"
TEXT_COL: str = "text"
AUDIO_DURATION_COL: str = "input_length" 
MODEL_INPUT_COL: str = "input_features"
MODEL_TARGET_COL: str = "labels"
SAMPLE_ID_COL: str = "file_id"
TARGET_SAMPLE_RATE: int = 16000
DEVICE: str = "cpu"
FEA_EXTRACTOR: WhisperProcessor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="mandarin", task="transcribe"
)


def test_hf_datasetdict_load_audio_jsonl() -> None:
    datasets_dict: DatasetDict = F.dataset.hf_datasetdict_load_audio_jsonl(
        DEMO_JSONL_PATH, None, None,  
        audio_path_col=AUDIO_PATH_COL, 
        audio_duration_col=AUDIO_DURATION_COL
    )
    for sample in datasets_dict["train"]:
        assert(AUDIO_PATH_COL in sample)
        assert(AUDIO_DURATION_COL in sample)
        assert(
            sample[AUDIO_DURATION_COL] == F.io.audio_get_meta(sample[AUDIO_PATH_COL])["duration_sec"]
        )
