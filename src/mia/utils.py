# -*- coding: utf-8 -*-
# file: utils.py
# date: 2024-02-08


import os
import json
import re
import librosa
import soundfile as sf
from tqdm import tqdm
from numpy import ndarray
from typing import Dict, List, Tuple, Optional


from .struct import SubtitleChunk, SubtitleChunks
from .struct import AudioMetadata


def chunk_audio_with_subtitle_chunks(
    output_dir: str, audio_path: str, subtitle_chunks: SubtitleChunks
) -> List[AudioMetadata]:
    print("Chunking '%s'" % audio_path)
    out: List[AudioMetadata] = []

    audio_name: str = audio_path.split("/")[-1].split(".")[0]
    audio_fmt: str = audio_path.split("/")[-1].split(".")[1]
    
    audio: Optional[ndarray] = None
    sample_rate: int = -1
    audio, sample_rate = librosa.load(audio_path, sr=None)
    
    for i, subtitle_chunk in enumerate(tqdm(subtitle_chunks)):
        audio_metadata: AudioMetadata = AudioMetadata()
        
        file_path: str = os.path.join(
            output_dir, 
            "%s_part%i.%s" % (audio_name, i, audio_fmt)
        )
        if os.path.exists(file_path):
            print("Audio chunk '%s' already exists." % file_path)
        else: 
            start_sample_idx: int = int(
                round(subtitle_chunk.start_in_second * sample_rate, 0)
            )
            end_sample_idx: int = int(
                round(subtitle_chunk.end_in_second * sample_rate, 0)
            )
            segment: ndarray = audio[start_sample_idx:end_sample_idx]

            sf.write(file_path, segment, sample_rate)
            #print("Finished saving chunk audio '%s'" % file_path)

        audio_metadata.transcript = subtitle_chunk.subtitle
        audio_metadata.path = file_path
        out.append(audio_metadata)

    return out


def json_objs2jsonl_file(path: str, json_objs: List[Dict]) -> str:
    file = open(path, "w")
    for record in json_objs:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.close()
    return path


def jsonl_file2json_objs(path: str) -> List[Dict]:
    return [
        json.loads(x) for x in open(path, "r").read().split("\n") 
        if x not in {""}
    ]


def split_text_by_chinese_punctuation(sentence):
    # Define Chinese punctuation marks
    chinese_punctuation = '！？｡。，：；、'

    # Use regular expression to split sentence by Chinese punctuation marks
    split_sentences = re.split(r'([' + chinese_punctuation + '])', sentence)

    # Remove empty strings and punctuation marks from the list
    split_sentences = [s for s in split_sentences if s and s not in chinese_punctuation]

    return split_sentences


def remove_punctuations_alphabets(input_string):
    # Remove Chinese and English punctuations
    chinese_punctuations = '【】（）《》“”‘’：“”'
    english_punctuations = r'''()-[]{}:'"\<>/@#$%^&*_~+*-'''
    punctuations_pattern = f"[{re.escape(chinese_punctuations)}{re.escape(english_punctuations)}]"

    # Remove alphabets and numbers
    #alphanum_pattern = r'[A-Za-z0-9]'
    alphanum_pattern = r'[A-Za-z]'

    # Combine patterns
    combined_pattern = f'{punctuations_pattern}|{alphanum_pattern}'

    # Remove specified characters using regex
    result = re.sub(combined_pattern, '', input_string)

    return result
