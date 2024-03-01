# -*- coding: utf-8 -*-
# file: argumentation.py
# date: 2024-02-28


import random as rd
import augly.audio as audaugs
from typing import List, Union
from torch import Tensor
from augly.audio import Compose, OneOf
from torchaudio.transforms import FrequencyMasking, TimeMasking



AUGLY_TRANSFORMS: Compose = audaugs.Compose([
    #audaugs.Clip(duration_factor=0.25),
    audaugs.AddBackgroundNoise(p=0.5),
    audaugs.ChangeVolume(volume_db=10.0, p=0.5),
    audaugs.OneOf(
        [audaugs.Speed(factor=3.0), audaugs.TimeStretch(rate=3.0)], 
        p=0.5
    ),
])


def spec_argument(
    spec: Union[List, Tensor],
    freq_before_time_axis: bool=True, 
    freq_masking_prob: float=1.0, 
    freq_max_masking_ratio: int=0.1, 
    time_masking_prob: float=1.0, 
    time_max_masking_ratio: int=0.1
) -> List:
    if isinstance(spec, list):
        spec = Tensor(spec)
    if len(spec.shape) not in {2, 3}:
        raise Exception("Dim error")
    if len(spec.shape) == 2: 
        spec = spec.reshape(-1, spec.shape[0], spec.shape[1])
    if not freq_before_time_axis:
        spec = spec.reshape(-1, spec.shape[1], spec.shape[2])

    freq_dim: int = spec.shape[1] 
    time_dim: int = spec.shape[2]
    freq_max_masking_len: int = int(freq_dim * freq_max_masking_ratio)
    time_max_masking_len: int = int(time_dim * time_max_masking_ratio)

    if rd.random() < freq_masking_prob:
        freq_masking: FrequencyMasking = FrequencyMasking(
            freq_mask_param=freq_max_masking_len
        )
        spec = freq_masking(spec)

    if rd.random() < time_masking_prob:
        time_masking: TimeMasking = TimeMasking(
            time_mask_param=time_max_masking_len
        )
        spec = time_masking(spec)

    return spec.tolist()



