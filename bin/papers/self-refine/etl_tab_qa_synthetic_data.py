# -*- coding: utf-8 -*-
# file: etl_tab_qa_synthetic_data.py
# date: 2025-04-16


import os
import sys
import pdb
import json
import asyncio
import traceback
import duckdb
from pydantic import BaseModel
from typing import (
    Dict, 
    List, 
    Optional, 
    Annotated, 
    Literal, 
    Callable, 
    Type, 
    Literal, 
    Coroutine
)
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from pandas import DataFrame
from langchain.chat_models import init_chat_model
from tqdm import tqdm


EXAMPLE0: str = \
"""
Vital Signs on 2020-05-14: 
Body temperature is 35.3. Heart rate is 93. Respiratory rate is 21. Oxygen saturature is 96. Systolic pressure is 130. Diastolic pressure is 067.

Vital Signs on 2020-05-15:
...

Vital Signs on 2020-05-13:
...

Vital Signs on 2020-05-20: 
...
""".strip("\n")     



USER_PROMPT_DATA_GEN: str = \
"""
You need following your task based on following information.

## Your Role and Task
Your task is to randomly generate a free-text which contains 
3 to 10 records which are convertable to table rows.

## Example Source Text
__EXAMPLE__
""".strip("\n")



USER_PROMPT_QUERY_GEN: str = \
"""
Based on your generated data which is convertable to table, 
please randomly generate a question which can be solved or answered 
by a SQL query's result.

You must make the generated questions have a strong randomness, 
and make the potential SQL solution need different types of operation.

You can also fake some contexts (like current datetime) for your 
question.

Only return your result without any explanation else.
"""


EXAMPLES: List[str] = [EXAMPLE0]


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]



async def run(
    llm: BaseChatModel, 
    example: str,
    user_prompt_data_gen: str=USER_PROMPT_DATA_GEN, 
    user_prompt_query_gen: str=USER_PROMPT_QUERY_GEN
) -> Dict[str, str]:
    msgs: List[AnyMessage] = []
    user_data_gen: HumanMessage = HumanMessage(
        content=USER_PROMPT_DATA_GEN.replace(
            "__EXAMPLE__", example
        )
    )
    msgs.append(user_data_gen)
    ai_data_gen: AIMessage = await llm.ainvoke(msgs)
    msgs.append(ai_data_gen)
    user_query_gen: HumanMessage = HumanMessage(
        content=USER_PROMPT_QUERY_GEN
    )
    msgs.append(user_query_gen)
    ai_query_gen: AIMessage = await llm.ainvoke(msgs, temperature=0.9) 
    return {
        "src_data": ai_data_gen.content, 
        "query_task": ai_query_gen.content
    }


async def run_n_times(
    llm: BaseChatModel,
    example: str,
    user_prompt_data_gen: str=USER_PROMPT_DATA_GEN,
    user_prompt_query_gen: str=USER_PROMPT_QUERY_GEN,
    n: int=8,
    batch: int=4
) -> List[Dict]:
    out: List[Dict] = []
    while len(out) < n:
        batch = min(batch, n - len(out))
        tasks: List[Coroutine] = [
            run(llm, example, user_prompt_data_gen, user_prompt_query_gen)
            for i in range(batch)
        ]
        out += (await asyncio.gather(*tasks))
    return out


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    out_data_path: str = configs["out_data_path"]
    llm_configs: Dict = configs["llm"]
    langchain_init(configs["langchain"])
    
    llm: BaseChatModel = init_chat_model(**llm_configs)
    out_file = open(out_data_path, "w")
    for i in tqdm(configs["example_ids"]):
        results: List[Dict[str, str]] = await run_n_times(
            llm, 
            EXAMPLES[i],
            n=configs["sample_size_per_example"]
        )
        for row in results:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_file.close()
    print("Results are dumped to '%s'" % out_data_path)
    return


if __name__ == "__main__":
    asyncio.run(main())
