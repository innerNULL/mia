# -*- coding: utf-8 -*-
# file: struct.py
# date: 2024-02-07


import json
from typing import Dict, List


class SubtitleChunk:
    def __init__(self):
        self.start_time: str = ""
        self.end_time: str = ""
        self.start_in_second: float = 0.0
        self.end_in_second: float = 0.0
        self.subtitle: str = ""


SubtitleChunks = List[SubtitleChunk]


class AudioMetadata:
    def __init__(self):
        self.path: str = ""
        self.transcript: str = ""


def subtitle_chunk_to_json_obj(chunk: SubtitleChunk) -> Dict:
    return {
        "start_time": chunk.start_time, 
        "end_time": chunk.end_time, 
        "start_in_second": chunk.start_in_second, 
        "end_in_second": chunk.end_in_second, 
        "subtitle": chunk.subtitle
    }


def subtitle_chunk_print(chunk: SubtitleChunk) -> None:
    print(subtitle_chunk_to_json_obj(chunk))


def subtitle_chunk_new(
    start_time: str, end_time: str, subtitle: str
) -> SubtitleChunk:
    """
    Args:
        start_time: In format like '01:36:34.752' 
        end_time: Same format as `start_time`
        subtitle: Text of subtitle
    """
    out: SubtitleChunk = SubtitleChunk()
    out.start_time = start_time
    out.end_time = end_time
    out.subtitle = subtitle
    
    start_hour: float = float(start_time.split(":")[0])
    start_min: float = float(start_time.split(":")[1])
    start_second: float = float(start_time.split(":")[2])
    end_hour: float = float(end_time.split(":")[0])
    end_min: float = float(end_time.split(":")[1])
    end_second: float = float(end_time.split(":")[2])

    out.start_in_second = start_hour * 60 * 60 + start_min * 60 + start_second
    out.end_in_second = end_hour * 60 * 60 + end_min * 60 + end_second
    return out


def subtitle_chunks_new_with_vvt(path: str) -> List[SubtitleChunk]:
    return None


def subtitle_chunks_merge(
    chunks: List[SubtitleChunk], max_gap: float=0
) -> List[SubtitleChunk]:
    out: List[SubtitleChunk] = []
    for curr_chunk in chunks:
        if len(out) == 0:
            out.append(curr_chunk)
            continue

        curr_start_in_second: float = curr_chunk.start_in_second
        curr_end_in_second: float = curr_chunk.end_in_second
        prev_start_in_second: float = out[-1].start_in_second
        prev_end_in_second: float = out[-1].end_in_second

        gap: float = curr_start_in_second - prev_end_in_second
        if gap <= max_gap:
            out[-1].end_in_second = curr_end_in_second
            out[-1].subtitle += (" " + curr_chunk.subtitle)
        else:
            out.append(curr_chunk)
    
    return out


def audio_metadata_to_json_obj(audio: AudioMetadata) -> Dict:
    return {
        "transcript": audio.transcript, "path": audio.path
    }

