# -*- coding: utf-8 -*-
# file: self_refine4tab_understanding.py
# date: 2025-04-14
"""
# Fake Data Generation Prompt
Please finish your task based on following instructions.

## Role
You're a fack data generator.

## Task
I want to build a multi-agents based table understanding system, which contains following steps:
* Parsing given free text into structured JSON lines format
* Based on user's requirement, generate DuckDB SQL to query from structured data
* Using self-refine mechanisom to refine generated SQL
* Run SQL to generated final result

Your need generate a fake dataset for me to development and debugging system.

## Output Format
JSON lines of fake samples. Each sample must contains following fields:
* src_data (string): The data which's convertable to structured data via LLM
* query_task (string)

Make sure each JSON must be in single line.
"""


import os
import sys
import pdb
import json
import asyncio
import traceback
import duckdb
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated, Literal, Callable, Type, Literal
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
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from tqdm import tqdm


DATA_PARSER_SYS_PROMPT: str = \
"""
You need finish your task based on following descriptions.

## Task
You're working with a free-text (source data) which is potential convertiable to structure data.
Your task is to:
* Locate all the data which are the source of user's instruction. 
* Convert all source data into a list of JSON objects which are sharing same schema.

## Output Format
The output should be and only be a list of JSON objects.
All JSON object must share same schema.
Following is an example:
[
  {...},
  {...}
]
If it's impossible to structurize the given data into JSON rows, then just return 'N/A'.
Your output must be enclosed with <answer> and </answer>.    
""".strip("\n")


CODE_GEN_SYS_PROMPT: str = \
"""
You need finish your task based on following information. 

# Your Role and Task
Your're a SQL programmer, you need write DuckDB SQL with:
* Data in the format of a list of JSON objects 
* A query requirement 

Your target is to write DuckDB SQL to satisfy the query requirement.

# Data
{data}

You can suppose the above data would be saved into a `pandas.DataFrame` 
named "source_data", and you need write DuckDB SQL to query from it.

# Query Requirement
{query_requirement}

# Output Format
You must only return your SQL code without anything else, 
and make it be enclosed with <answer> and </answer>. 
""".strip("\n")


class State(BaseModel):
    llms: Dict[str, BaseChatModel]
    data_parser_msgs: List[AnyMessage] = []
    code_gen_msgs: List[AnyMessage] = []
    refine_rounds: int = 0
    src_data: Optional[str] = None
    query_task: Optional[str] = None
    parsed_data: Optional[List[Dict]] = None
    sql: Optional[str] = None
    sql_passed: Optional[bool] = None
    final_result: Optional[bool] = None
    data_parsing_sys_prompt: str = DATA_PARSER_SYS_PROMPT
    code_gen_sys_prompt: str = CODE_GEN_SYS_PROMPT


class Config(BaseModel):
    llm_data_parser: str
    llm_code_gen: str
    llm_code_feedback: str
    llm_code_refine: str


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]


def config_init(
    agent_configs: Dict
) -> Config:
    out: Config = Config(
        llm_data_parser=agent_configs["data_parser"]["llm"],
        llm_code_gen=agent_configs["code_gen"]["llm"],
        llm_code_feedback=agent_configs["code_feedback"]["llm"],
        llm_code_refine=agent_configs["code_refine"]["llm"]
    )
    return out


def tag_extractor(data: str, tag_start: str, tag_end: str) -> str:
    start_idx: int = data.index(tag_start) + len(tag_start)
    end_idx: int = data.index(tag_end)
    return data[start_idx:end_idx]


def sql_run(sql: str, data: List[Dict]) -> Dict[str, str]:
    """
    This is function will:
    * Convert `data` to `pandas.DataFrame`.
    * Run `sql` which is a DuckDB query on `pandas.DataFrame` based on `data`
    """
    source_data: DataFrame = DataFrame(data)
    result: Dict = {"result": None, "err": None}
    try:
        result["result"] = duckdb.query(sql).df().to_markdown(index=False)
    except Exception as e:
        result["err"] = traceback.format_exc()
    return result


async def agent_data_parser(state: State, config: Dict) -> Dict:
    cus_conf: Dict = config["configurable"]
    llm_name: str = cus_conf["llm_data_parser"]
    llm: BaseChatModel = state.llms[llm_name]
    data_parsing_sys_prompt: str = state.data_parsing_sys_prompt      
    src_data: str = state.src_data
    query_task: str = state.query_task
    msgs: List[AnyMessage] = []
    sys_msg: SystemMessage = SystemMessage(content=data_parsing_sys_prompt)
    user_msg: HumanMessage = HumanMessage(
        content="# Instruction\n{}\n\n# Source Data\n{}".format(
            query_task, 
            src_data
        )
    )
    msgs.append(sys_msg)
    msgs.append(user_msg)
    resp: AIMessage = await llm.ainvoke(msgs)
    msgs.append(resp)

    parsed_data: Optional[List[Dict]] = None
    while parsed_data is None:
        try:
            parsed_data = json.loads(
                tag_extractor(msgs[-1].content, "<answer>", "</answer>")
            )
        except Exception as e:
            err: str = traceback.format_exc()
            err_msg: HumanMessage = HumanMessage(
                content=(
                    "Something wrong happened when extracting content "
                    "enclosed by <answer> and </answer> and then parsing it to JSON, "
                    "please fix your answer. "
                    "Potentiall problems are: \n"
                    "* You didn't enclose your answer with <answer> and </answer>\n"
                    "& The list of JSON objects are illegal and not able to be parsed."
                )
            )
            msgs.append(err_msg)
            msgs.append((await llm.ainvoke(msgs)))
            
    return {"data_parser_msgs": msgs, "parsed_data": parsed_data}


async def agent_code_gen(
    state: State, 
    config: Config
) -> Dict:
    cus_conf: Dict = config["configurable"]
    llm_name: str = cus_conf["llm_code_gen"]
    llm: BaseChatModel = state.llms[llm_name]
    msgs: List[AnyMessage] = state.code_gen_msgs
    sys_msg: SystemMessage = HumanMessage(
        content=state.code_gen_sys_prompt.format(
            data=json.dumps(state.parsed_data, indent=2),
            query_requirement=state.query_task
        )
    )
    msgs.append(sys_msg)
    resp: AIMessage = await llm.ainvoke(msgs)
    msgs.append(resp)
    sql: str = tag_extractor(resp.content, "<answer>", "</answer>")
    passed: bool = False
    result: Optional[Dict[str, str]] = None
    for i in range(state.refine_rounds):
        result = sql_run(sql, state.parsed_data)
        if result["err"] is not None:
            err: str = result["err"]
            err_msg: HumanMessage = HumanMessage(
                content="Please fix your code based on following error:\n%s" % err
            )
            msgs.append(err_msg)
            msgs.append((await llm.ainvoke(msgs)))
            sql = tag_extractor(msgs[-1].content, "<answer>", "</answer>")
            continue
        passed = True
    return {
        "code_gen_msgs": msgs, 
        "sql": sql if passed else None, 
        "sql_passed": passed,
        "final_result": result["result"]
    }


async def router_data_parser(
    state: State, 
    config: Dict
) -> Literal["agent_code_gen", END]:
    raw_parsed_data: str = state.data_parser_msgs[-1].content
    if "N/A" in raw_parsed_data:
        return END
    else:
        return "agent_code_gen"


def graph_build(
    state: Type, 
    config: Type, 
    agent_data_parser: Callable,
    agent_code_gen: Callable, 
    router_data_parser: Callable
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(state, config_schema=config)
    builder.add_node("agent_data_parser", agent_data_parser)
    builder.add_node("agent_code_gen", agent_code_gen)
    builder.add_edge(START, "agent_data_parser")
    builder.add_conditional_edges("agent_data_parser", router_data_parser)
    builder.add_edge("agent_code_gen", END)
    graph: CompiledStateGraph = builder.compile()
    return graph


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    llm_configs: Dict = configs["llms"]
    agent_configs: Dict = configs["agents"]
    in_data_path: str = configs["in_data_path"]
    src_data_col: str = configs["src_data_col"]
    query_task_col: str = configs["query_task_col"]

    langchain_init(configs["langchain"])
    llms: Dict[str, BaseChatModel] = {
        x["model"]: init_chat_model(**x) for x in llm_configs
    }
    samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ][:configs["max_sample_size"]]
    graph: CompiledStateGraph = graph_build(
        State, 
        Config,
        agent_data_parser,
        agent_code_gen,
        router_data_parser
    )
    graph_conf: Dict = {
        "configurable": config_init(agent_configs).dict() 
    }
    for sample in tqdm(samples):
        inputs: Dict = {
            "llms": llms,
            "src_data": sample[src_data_col],
            "query_task": sample[query_task_col],
            "refine_rounds": 5
        }
        result: State = await graph.ainvoke(inputs, config=graph_conf)
        pdb.set_trace()
    pdb.set_trace()
    return


if __name__ == "__main__":
    asyncio.run(main())

