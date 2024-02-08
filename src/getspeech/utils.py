# -*- coding: utf-8 -*-
# file: utils.py
# date: 2024-02-08


import os
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
        
        start_sample_idx: int = int(round(subtitle_chunk.start_in_second * sample_rate, 0))
        end_sample_idx: int = int(round(subtitle_chunk.end_in_second * sample_rate, 0))
        segment: ndarray = audio[start_sample_idx:end_sample_idx]
        
        file_path: str = os.path.join(
            output_dir, 
            "%s_part%i.%s" % (audio_name, i, audio_fmt)
        )

        sf.write(file_path, segment, sample_rate)
        #print("Finished saving chunk audio '%s'" % file_path)

        audio_metadata.transcript = subtitle_chunk.subtitle
        audio_metadata.path = file_path
        out.append(audio_metadata)

    return out

