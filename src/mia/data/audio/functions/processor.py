# -*- coding: utf-8 -*-
# file: processor.py
# date: 2024-03-08


from opencc import OpenCC


def text_force_simplified_chinese(text: str, lang: str="") -> str:
    if lang.lower() not in {"mandarin", "zh-cn", "zh-tw", "zh"}:
        return text
    return OpenCC("tw2s.json").convert(text)
