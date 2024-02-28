# -*- coding: utf-8 -*-
# file: argumentation.py
# date: 2024-02-28


import augly.audio as audaugs
from augly.audio import Compose, OneOf


AUGLY_TRANSFORMS: Compose = audaugs.Compose([
    #audaugs.Clip(duration_factor=0.25),
    audaugs.AddBackgroundNoise(p=0.5),
    audaugs.ChangeVolume(volume_db=10.0, p=0.5),
    audaugs.OneOf(
        [audaugs.Speed(factor=3.0), audaugs.TimeStretch(rate=3.0)], 
        p=0.5
    ),
])


