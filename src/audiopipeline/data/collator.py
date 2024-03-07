# -*- coding: utf-8 -*-
# file: collator.py
# date: 2024-03-01


import torch
from torch import Tensor
from typing import Any, Optional, Dict, List, Union

from .argumentation import spec_argument
from .functions import josnl_record2train_sample


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

    def spec_argumentation(self, samples: List[Dict]) -> List[Dict]:
        if not self.spec_argument:
            return samples
        
        #print("Running Spec-Argument")
        for sample in samples:
            sample[self.model_input_col] = spec_argument(
                sample[self.model_input_col],
                freq_before_time_axis=True,
                freq_masking_prob=self.freq_masking_prob, 
                freq_max_masking_ratio=self.freq_max_masking_ratio, 
                time_masking_prob=self.time_masking_prob, 
                time_max_masking_ratio=self.time_max_masking_ratio
            )
        return samples

    def __call__(
        self, features: List[Dict[str, Union[List[int], Tensor]]]
    ) -> Dict[str, Tensor]:
        features = self.spec_argumentation(features) 
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


class DataCollatorSpeechSeq2SeqWithPaddingV1:
    def __init__(self, 
        processor: Any, 
        tokenizer: Any=None, 
        path_col: str="path", 
        text_col: str="text",
        audio_duration_col: str="input_length",
        model_input_col: str="input_features", 
        model_label_col: str="labels", 
        sample_id_col: str="",
        target_sample_rate: int=16000, 
        spec_argument: bool=True,
        freq_masking_prob: float=0.7, 
        freq_max_masking_ratio: float=0.1,
        time_masking_prob: float=0.7, 
        time_max_masking_ratio: float=0.1
    ):
        self.processor: Any = processor
        self.tokenizer: Any = self.processor.tokenizer if tokenizer is None else tokenizer
        self.path_col: str = path_col
        self.text_col: str = text_col
        self.audio_duration_col: str = audio_duration_col
        self.model_input_col: str = model_input_col
        self.model_label_col: str = model_label_col
        self.sample_id_col: str = sample_id_col
        self.target_sample_rate: int = target_sample_rate
        self.spec_argument: bool = spec_argument
        self.freq_masking_prob: float = freq_masking_prob
        self.freq_max_masking_ratio: float = freq_max_masking_ratio
        self.time_masking_prob: float = time_masking_prob
        self.time_max_masking_ratio: float = time_max_masking_ratio

    def __call__(self, jsonl_samples: List[Dict]) -> Dict[str, Tensor]:
        train_samples: List[Dict] = [
            josnl_record2train_sample(
                x, self.processor, 
                path_col=self.path_col, text_col=self.text_col, 
                model_input_col=self.model_input_col, 
                model_target_col=self.model_label_col, 
                audio_duration_col=self.audio_duration_col, 
                target_sample_rate=self.target_sample_rate
            ) for x in jsonl_samples
        ]

        input_features: List[Dict[str, Union[List[float], Tensor]]] = [
            {self.model_input_col: sample[self.model_input_col].tolist()[0]} 
            for sample in train_samples
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [
            {"input_ids": sample[self.model_label_col]} for sample in train_samples
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

        if self.sample_id_col not in {""}:
            batch[self.sample_id_col] = torch.tensor(
                [x[self.sample_id_col] for x in jsonl_samples], dtype=torch.int32
            ).reshape(len(jsonl_samples), 1)

        return batch

