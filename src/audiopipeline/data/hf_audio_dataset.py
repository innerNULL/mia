# -*- coding: utf-8 -*-
# file: hf_audio_dataset.py
# date: 2024-02-26


import pdb
import copy
import opencc
import numpy as np
from typing import Dict, List
from datasets import load_dataset
from datasets import DatasetDict, Dataset
from datasets import Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Trainer, TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl

from . import argumentation


class HfAudioDataset:
    def __init__(self, 
        train_data_path: str, dev_data_path: str, test_data_path: str,
        processor: WhisperProcessor,
        sampling_rate: int=16000, lang: str="zh-TW", 
        audio_path_col: str="path",
        text_col: str="text", 
        audio_col: str="audio",
        duration_col: str="input_length",
        min_duration: int=0, max_duration: int=30, 
        max_label_len: int=448,
        num_proc: int=4
    ):
        self.processor: WhisperProcessor = processor
        self.train_data_path: str = train_data_path
        self.dev_data_path: str = dev_data_path
        self.test_data_path: str = test_data_path
        self.sampling_rate: int = sampling_rate
        self.lang: str = lang
        self.audio_path_col: str = audio_path_col
        self.text_col: str = text_col
        self.audio_col: str = audio_col
        self.duration_col: str = duration_col
        self.min_duration: int = min_duration
        self.max_duration: int = max_duration
        self.max_label_len: int = max_label_len
        self.num_proc: int = num_proc

        self.static_datasets: DatasetDict = None

        self.init_static_datasets()

    def init_static_datasets(self) -> None:
        datasets: DatasetDict = datasetdict_load_jsonl(
            self.train_data_path, 
            self.dev_data_path, self.test_data_path
        )
        
        for split in datasets:
            dataset: Dataset = datasets[split]
            dataset = dataset_load_audio(
                dataset, 
                sampling_rate=self.sampling_rate, 
                audio_path_col=self.audio_path_col, audio_col=self.audio_col
            )
            dataset.cleanup_cache_files()
            datasets[split] = dataset

        self.static_datasets = datasets

    def get_final_datasets(self, argumentation_splits: List[str]="train") -> DatasetDict:
        datasets: DatasetDict = copy.deepcopy(self.static_datasets)
        for split in datasets:
            print("Building final %s dataset" % split)
            dataset: Dataset = datasets[split].shuffle()
            dataset.cleanup_cache_files()

            if split in argumentation_splits:
                dataset = dataset_audio_time_domain_argumentation(
                    dataset, self.audio_col, self.sampling_rate, 
                    num_proc=self.num_proc
                )
                dataset.cleanup_cache_files()

            dataset = dataset_raw_transcript_processor(
                dataset, text_col=self.text_col, lang=self.lang, num_proc=self.num_proc
            )
            dataset.cleanup_cache_files()

            dataset = dataset_run_hf_processor(
                dataset, 
                self.processor, self.audio_col, self.text_col, self.duration_col, 
                num_proc=self.num_proc
            )
            dataset.cleanup_cache_files()

            if split in argumentation_splits:
                dataset = dataset_audio_freq_domain_argumentation(dataset)
                dataset.cleanup_cache_files()

            dataset = dataset_filter(
                dataset, 
                self.min_duration, self.max_duration, self.max_label_len,
                duration_col=self.duration_col
            )
            dataset.cleanup_cache_files()

            datasets[split] = dataset
        
        return datasets


class DataArgumentationCallback(TrainerCallback):
    def __init__(self, dataset: HfAudioDataset, trainer: Trainer):
        self.dataset: HfAudioDataset = dataset
        self.trainer: Trainer = trainer

    def on_epoch_begin(
        self, 
        args: TrainingArguments, state: TrainerState, control: TrainerControl, 
        **kwargs
    ):
        if state.epoch == 0 and self.trainer.train_dataset is not None:
            pass
        else:
            print("Re-constructing training dataset")
            self.trainer.train_dataset = None
            self.trainer.train_dataset = self.dataset.get_final_datasets()["train"]


def datasetdict_load_jsonl(
    train_data_path: str, dev_data_path: str, test_data_path: str
) -> DatasetDict:
    print("Runing dataset dict JSONL loader")
    dataset: DatasetDict = DatasetDict()

    if train_data_path is not None:
        dataset["train"] = load_dataset("json", data_files=train_data_path)["train"]
    if dev_data_path is not None:
        dataset["validation"] = load_dataset("json", data_files=dev_data_path)["train"]
    if test_data_path is not None:
        dataset["test"] = load_dataset("json", data_files=test_data_path)["train"]
    
    return dataset


def dataset_load_audio(
    jsonl_dataset: Dataset, 
    sampling_rate: int=16000, audio_path_col: str="path", audio_col: str="audio"
) -> Dataset:
    print("Running dataset audio loader")
    dataset: Dataset = jsonl_dataset.add_column(
        audio_col, jsonl_dataset[audio_path_col]
    )
    dataset = dataset.cast_column(
        audio_col, Audio(sampling_rate=sampling_rate)
    )
    return dataset


def sample_audio_time_domain_argumentation(
    sample: Dict, audio_col: str, sample_rate: int
) -> Dict:
    sample[audio_col]["array"] = argumentation.AUGLY_TRANSFORMS(
        sample[audio_col]["array"], sample_rate
    )[0]
    if sample[audio_col]["array"].dtype == np.dtype("float64"):
        sample[audio_col]["array"] = np.float32(sample[audio_col]["array"])
    return sample


def dataset_audio_time_domain_argumentation(
    audio_dataset: Dataset, audio_col: str, sample_rate: int, 
    num_proc: int=4
) -> Dataset:
    print("Running time domain data argumentation")
    return audio_dataset.map(
        sample_audio_time_domain_argumentation, 
        fn_kwargs={"audio_col": audio_col, "sample_rate": sample_rate},
        num_proc=num_proc
    )


def dataset_raw_transcript_processor(
    audio_dataset: Dataset, text_col: str, lang: str,
    num_proc: int=4
) -> Dataset:
    print("Running dataset raw text pre-processing")
    dataset: Dataset = audio_dataset
    if lang in {"zh-TW", "mandarin"}:
        print("Converting %s to simplified Chinese" % lang)
        def convert_text(example: Dict, text_col: str) -> Dict:
            converter: opencc.OpenCC = opencc.OpenCC('tw2s.json')
            example[text_col] = converter.convert(example[text_col])
            return example
        dataset = dataset.map(
            convert_text, fn_kwargs={"text_col": text_col}, 
            num_proc=num_proc
        )
    return dataset


def sample_hf_processor(
    sample: Dict, processor: WhisperProcessor, audio_col: str, text_col: str, 
    duration_col: str="input_length"
) -> Dict:
    audio: Dict = sample[audio_col]
    output = processor(
        audio=audio["array"], sampling_rate=audio["sampling_rate"], 
        text=sample[text_col]
    )
    output[duration_col] = len(audio["array"]) / audio["sampling_rate"]
    return output


def dataset_run_hf_processor(
    audio_dataset: Dataset, processor: WhisperProcessor, 
    audio_col: str, text_col: str, 
    duration_col: str="input_length", 
    num_proc: int=4
) -> Dataset:
    print("Running dataset HuggingFace processor")
    dataset: Dataset = audio_dataset.map(
        sample_hf_processor, 
        fn_kwargs={
            "processor": processor, 
            "audio_col": audio_col, "text_col": text_col, 
            "duration_col": duration_col
        },
        num_proc=num_proc
    )
    return dataset


def dataset_audio_freq_domain_argumentation(spectrum_dataset: Dataset) -> Dataset:
    print("Running dataset frequent domain data argumentation")
    return spectrum_dataset


def sample_filter_flag(
    audio_duration: int, label_tokens: int, 
    min_duration: int, max_duration: int, max_label_len: int,
) -> bool:
    if audio_duration <= min_duration:
        return False
    if audio_duration >= max_duration:
        return False
    if len(label_tokens) >= max_label_len:
        return False
    return True


def dataset_filter(
    hf_processed_dataset: Dataset, 
    min_duration: int, max_duration: int, max_label_len: int,
    duration_col: str="input_length"
) -> Dataset:
    print("Running dataset filter")
    dataset: Dataset = hf_processed_dataset.filter(
        sample_filter_flag, 
        input_columns=[duration_col, "labels"],
        fn_kwargs={
            "min_duration": min_duration, 
            "max_duration": max_duration, 
            "max_label_len": max_label_len
        }
    ) 
    return dataset
