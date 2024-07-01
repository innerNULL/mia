# -*- coding: utf-8 -*-
# file: lib.py
# date: 2024-06-29


from typing import Union, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOllama


def init_llm_client(
    llm_engine_type: str, 
    llm_engine_api: Optional[str],
    llm: str
) -> Optional[Union[BaseLanguageModel]]:
    if llm_engine_type in {"ollama", "Ollama"}:
        return ChatOllama(
            model=llm,
            base_url=llm_engine_api,
            temperature=0, 
            top_k=1
        )
    else:
        return None
    
