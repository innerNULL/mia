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
    freq_axis: int=1, time_axis: int=2,
    freq_masking_prob: float=1.0, 
    freq_max_masking_ratio: int=0.1, 
    time_masking_prob: float=1.0, 
    time_max_masking_ratio: int=0.1
) -> List:
    if isinstance(spec, list):
        spec = Tensor(spec)

    freq_dim: int = spec.shape[freq_axis] 
    time_dim: int = spec.shape[time_axis]
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



