# -*- coding: utf-8 -*-
# file: hf_audio_dataset_dev.py
# date: 2024-02-27
#
# Usage:
# python dev/audiopipeline/data/collator_dev.py ./common_voice_16_1_dev.jsonl


import pdb
import sys
import os
from torch import Tensor
from datasets import set_caching_enabled
from typing import Dict, List
from datasets import DatasetDict, Dataset
from transformers import WhisperProcessor

#sys.path.append(
#    os.path.join(os.path.dirname(__file__), "../../../src")
#)
from audiopipeline.data import hf_audio_dataset
from audiopipeline.data.hf_audio_dataset import HfAudioDataset
from audiopipeline.data.collator import HfDataCollatorSpeechSeq2SeqWithPadding


LANG: str = "zh-TW"
TRANSCRIPT_COL: str = "text"
AUDIO_COL: str = "audio"
DURATION_COL: str = "input_length"


if __name__ == "__main__":
    jsonl_data_path: str = sys.argv[1]

    set_caching_enabled(False)

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="mandarin", task="transcribe"
    )

    dataset: HfAudioDataset = HfAudioDataset(
        jsonl_data_path, jsonl_data_path, jsonl_data_path, processor, 
        sampling_rate=16000, 
        lang=LANG, 
        audio_path_col="path", text_col=TRANSCRIPT_COL, 
        audio_col=AUDIO_COL, duration_col=DURATION_COL,
        max_duration=3, 
        waveform_argument_splits=[]
    )
    collator: HfDataCollatorSpeechSeq2SeqWithPadding = \
        HfDataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, tokenizer=None, 
            model_input_col="input_features", model_label_col="labels", 
            spec_argument=True, 
            freq_masking_prob=0.7, freq_max_masking_ratio=0.1, 
            time_masking_prob=0.7, time_max_masking_ratio=0.1
        )

    final_train_data: List[Dict] = [
        x for x in dataset.get_final_datasets()["train"]
    ]
    final_train_batch: List[Dict] = final_train_data[:32]
    train_inputs: Dict[str, Tensor] = collator(final_train_batch)

    pdb.set_trace()
