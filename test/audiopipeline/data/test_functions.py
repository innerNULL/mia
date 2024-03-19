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


def test_datasetdict_load_jsonl_0() -> None:
    datasets_dict: DatasetDict = F.datasetdict_load_jsonl(
        DEMO_JSONL_PATH, DEMO_JSONL_PATH, DEMO_JSONL_PATH
    )
    assert(AUDIO_PATH_COL in datasets_dict["train"][0])
    assert(TEXT_COL in datasets_dict["train"][0])


def test_datasetdict_load_jsonl_1() -> None:
    datasets_dict: DatasetDict = F.datasetdict_load_jsonl(
        DEMO_JSONL_PATH, DEMO_JSONL_PATH, DEMO_JSONL_PATH, 
        sample_id_col=SAMPLE_ID_COL
    )
    assert(SAMPLE_ID_COL in datasets_dict["train"][0])


def test_audio_file2model_inputs() -> None:
    audio_path: str = DEMO_JSONL_DATA[0][AUDIO_PATH_COL]
    model_input: Tensor = None
    audio_duration: int = -1 

    model_input, audio_duration = F.audio_file2model_inputs(
        audio_path, FEA_EXTRACTOR, TARGET_SAMPLE_RATE, DEVICE
    )
    assert(len(model_input.shape) == 3)
    assert(model_input.shape[2] == 3000)
    assert(model_input.shape[1] == 80)

    waveform, sr = librosa.load(audio_path)
    true_duration: int = librosa.get_duration(y=waveform, sr=sr)
    another_duration: int = F.io.audio_get_meta(audio_path)["duration_sec"]
    assert(round(true_duration, 1) == round(audio_duration, 1))
    assert(round(another_duration, 1) == round(audio_duration, 1))


def test_text2token_ids() -> None:
    text: str = DEMO_JSONL_DATA[0][TEXT_COL]
    output: Tensor = F.text2token_ids(text, FEA_EXTRACTOR)
    assert(isinstance(output, list))
    assert(isinstance(output[0], int))


def test_josnl_record2train_sample() -> None:
    jsonl_record: Dict = DEMO_JSONL_DATA[0]
    train_sample: Dict = F.josnl_record2train_sample(
        jsonl_record, FEA_EXTRACTOR,
        lang="mandarin",
        path_col=AUDIO_PATH_COL, 
        text_col=TEXT_COL,
        model_input_col=MODEL_INPUT_COL, 
        model_target_col=MODEL_TARGET_COL, 
        audio_duration_col=AUDIO_DURATION_COL,
        target_sample_rate=TARGET_SAMPLE_RATE, 
        device=DEVICE
    )

    assert(MODEL_INPUT_COL in train_sample)
    assert(MODEL_TARGET_COL in train_sample)
    assert(AUDIO_DURATION_COL in train_sample)
    
    decoded_text: str = FEA_EXTRACTOR.tokenizer.decode(
        train_sample[MODEL_TARGET_COL], skip_special_tokens=True
    )
    origin_text: str = jsonl_record[TEXT_COL]

    assert(OpenCC("tw2s.json").convert(decoded_text) == decoded_text)
    
