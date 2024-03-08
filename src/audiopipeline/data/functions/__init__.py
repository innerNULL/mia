# -*- coding: utf-8 -*-
# file: functions.py
# date: 2024-03-06


import pdb
import torch
import torchaudio
from datasets import load_dataset
from typing import Dict, Callable, Union, List, Tuple
from torch import Tensor
from datasets import DatasetDict, Dataset
from transformers import WhisperProcessor


def datasetdict_load_jsonl(
    train_data_path: str, dev_data_path: str, test_data_path: str, 
    sample_id_col: str=""
) -> DatasetDict:
    print("Running dataset dict JSONL loader")
    dataset: DatasetDict = DatasetDict()

    if train_data_path is not None:
        dataset["train"] = load_dataset("json", data_files=train_data_path)["train"]
    if dev_data_path is not None:
        dataset["validation"] = load_dataset("json", data_files=dev_data_path)["train"]
    if test_data_path is not None:
        dataset["test"] = load_dataset("json", data_files=test_data_path)["train"]
    
    if sample_id_col not in {""}:
        def _add_sample_id(sample: Dict, idx: int) -> Dict:
            sample[sample_id_col] = idx
            return sample

        for split in dataset:
            dataset[split] = dataset[split].map(_add_sample_id, with_indices=True)

    return dataset


def audio_file2model_inputs(
    path: str, fea_extractor: WhisperProcessor, 
    target_sample_rate: int=16000, device: str="cpu" 
) -> Tuple[Tensor, int]:
    waveform: Tensor = None
    sample_rate: int = -1
    waveform, sample_rate = torchaudio.load(path)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    duration_sec: int = waveform.shape[-1] / target_sample_rate
    inputs: Tensor = fea_extractor(
        waveform.squeeze(), sampling_rate=target_sample_rate, 
        return_tensors="pt"
    ).input_features.to(torch.device(device))
    return (inputs, duration_sec)


def text2token_ids(text: str, fea_extractor: WhisperProcessor) -> List[List[int]]:
    out: Tensor = fea_extractor(audio=None, text=text)["input_ids"]
    return out


def josnl_record2train_sample(
    jsonl_sample: Dict, 
    fea_extractor: WhisperProcessor,
    path_col: str="path", 
    text_col: str="text", 
    model_input_col: str="input_features", 
    model_target_col: str="labels", 
    audio_duration_col: str="input_length",
    target_sample_rate: int=16000, 
    device: str="cpu"
) -> Dict[str, Union[Tensor, int, str]]:
    output: Dict[str, Union[Tensor, int, str]] = {}

    output[text_col] = jsonl_sample[text_col] 
    
    output[model_input_col] = None
    output[audio_duration_col] = None
    output[model_input_col], output[audio_duration_col] = \
        audio_file2model_inputs(
            path=jsonl_sample[path_col], 
            fea_extractor=fea_extractor, 
            target_sample_rate=target_sample_rate,
            device=device
        ) 

    output[model_target_col] = text2token_ids(
        text=jsonl_sample[text_col],
        fea_extractor=fea_extractor,
    )
    return output
