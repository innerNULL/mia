# -*- coding: utf-8 -*-
# file: io.py
# date: 2024-03-08


import torchaudio
from typing import Union, Dict


def audio_get_meta(
    path: str, 
    audio_path_col: str="path",
    audio_duration_col: str="duration_sec"
) -> Dict[str, Union[str, int, float]]:
    metadata: Dict[str, Union[str, int, float]] = {}
    metadata[audio_path_col] = path

    waveform: Tensor = None
    sample_rate: int = -1
    waveform, sample_rate = torchaudio.load(path)
    
    duration_sec: int = waveform.shape[-1] / sample_rate
    metadata[audio_duration_col] = duration_sec

    return metadata
