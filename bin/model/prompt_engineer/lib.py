# -*- coding: utf-8 -*-
# file: lib.py
# date: 2024-06-29


import multiprocessing
from typing import Union, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig


def init_llm_client(
    llm_engine_type: str, 
    llm_engine_api: Optional[str],
    llm: str, 
    sampling: bool=True, 
    temperature: float=0.01, 
    top_k: int=1,
    local_model_path: str=""
) -> Optional[Union[BaseLanguageModel]]:
    if llm_engine_type in {"ollama", "Ollama"}:
        return ChatOllama(
            model=llm,
            base_url=llm_engine_api,
            temperature=temperature, 
            top_k=top_k
        )
    elif llm_engine_type in {"hf", "huggingface", "HuggingFace"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        llm_runner = HuggingFacePipeline.from_model_id(
            model_id=llm, 
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512, 
                "do_sample": sampling, 
                "temperature": temperature, 
                "top_k": top_k,
                "repetition_penalty": 1.03,
                "return_full_text": False
            },
            model_kwargs={
                "quantization_config": quant_config
            }
        )
        return ChatHuggingFace(llm=llm_runner)
    elif llm_engine_type in {"llama.cpp", "llamacpp"}:
        return ChatLlamaCpp(
            temperature=temperature,
            model_path=local_model_path,
            n_ctx=10000,
            n_gpu_layers=8,
            n_batch=300,
            max_tokens=512,
            n_threads=multiprocessing.cpu_count() - 1,
            repeat_penalty=1.03,
            #top_p=0.5,
            verbose=True,
        )
    else:
        return None
    
