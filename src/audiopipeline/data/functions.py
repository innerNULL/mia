# -*- coding: utf-8 -*-
# file: functions.py
# date: 2024-03-06


import pdb
import torch
import torchaudio
from typing import Dict, Callable
from torch import Tensor
from transformers import WhisperProcessor


def audio_file2model_inputs(
    path: str, fea_extractor: WhisperProcessor, 
    target_sample_rate: int=16000, device: str="cpu" 
) -> None:
    waveform: Tensor = None
    sample_rate: int = -1
    waveform, sample_rate = torchaudio.load(path)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    inputs: Tensor = fea_extractor(
        waveform.squeeze(), sampling_rate=target_sample_rate, 
        return_tensors="pt"
    ).input_features.to(torch.device(device))
    return inputs

