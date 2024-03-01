# -*- coding: utf-8 -*-
# file: collator.py
# date: 2024-03-01


import torch
from torch import Tensor
from typing import Any, Optional, Dict, List, Union

from .argumentation import spec_argument


class HfDataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, 
        processor: Any, 
        tokenizer: Any=None, 
        model_input_col: str="input_features", 
        model_label_col: str="labels",
        spec_argument: bool=True,
        freq_masking_prob: float=0.7, 
        freq_max_masking_ratio: float=0.1,
        time_masking_prob: float=0.7, 
        time_max_masking_ratio: float=0.1
    ):
        self.processor: Any = processor
        self.tokenizer: Any = self.processor.tokenizer if tokenizer is None else tokenizer
        self.model_input_col: str = model_input_col
        self.model_label_col: str = model_label_col
        self.spec_argument: bool = spec_argument
        self.freq_masking_prob: float = freq_masking_prob
        self.freq_max_masking_ratio: float = freq_max_masking_ratio
        self.time_masking_prob: float = time_masking_prob
        self.time_max_masking_ratio: float = time_max_masking_ratio

    def spec_argument(self, samples: List[Dict]) -> List[Dict]:
        if not self.spec_argument:
            return samples

        for sample in samples:
            sample[self.model_input_col] = spec_argument(
                example["input_features"],
                freq_axis=1, time_axis=2, 
                freq_masking_prob=freq_masking_prob, 
                freq_max_masking_ratio=self.freq_max_masking_ratio, 
                time_masking_prob=self.time_masking_prob, 
                time_max_masking_ratio=self.time_max_masking_ratio
            )
        return samples

    def __call__(
        self, features: List[Dict[str, Union[List[int], Tensor]]]
    ) -> Dict[str, Tensor]:
        input_features: List[Dict[str, Union[List[float], Tensor]]] = [
            {self.model_input_col: feature[self.model_input_col][0]} 
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [
            {"input_ids": feature[self.model_label_col]} for feature in features
        ]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
