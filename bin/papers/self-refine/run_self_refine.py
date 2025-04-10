# -*- coding: utf-8 -*-
# file: run_self_refine.py
# date: 2025-08-07


import os
import sys
import pdb
import json
import asyncio
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated, Literal, Callable
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from tqdm import tqdm


FEEDBACK_SYS_PROMPT: str = \
"""
You need finish your task based on following instructions.

# Role
Each time user will send you response of the instruction based on source document.

You need check if the given response has following problems:
* Hallucination: If anything in response can not be found or supported by source document.
* Instruction following: If the response is exactly following the given instruction.
* Redundancy: If there any room to make response be more concise.

# Instruction
__INSTRUCTION__

# Source Document
__SRC_DOC__

# Output Format
Return "No problem found." if nothing wrong.
Else return the name of problems you found and corresponding detailed reasons and citations.
"""


REFINE_SYS_PROMPT: str = \
"""
You need finish your task based on following instructions.

# Role
User will send you feedback of the response of a source document with specific instruction. 
Based on the feedback, source document and instruction, you need refine the response to eliminate the problems. 

# Source Document
__SRC_DOC__

# Instruction
__INSTRUCTION__

# Output Format
Your refined summary based on feedback and source document without any redundant explanations.
"""


class State(BaseModel):
    init_data: Dict
    msgs: List[AnyMessage] = []
    generator_msgs: List[AnyMessage] = []
    refine_msgs: List[AnyMessage] = []
    feedback_msgs: List[AnyMessage] = []
    remain_rounds: Optional[int] = None
    llm: Optional[BaseChatModel] = None


class Config(BaseModel):
    src_text_col: str
    init_gen_text_col: str
    instruction_col: str


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]


async def refine(state: State, config: Config) -> Dict:
    llm: BaseChatModel = state.llm
    conf: Dict = config["configurable"]
    msg: List[AnyMessage] = state.refine_msgs
    src_doc: str = state.init_data[conf["src_text_col"]]
    instruction: str = state.init_data[conf["instruction_col"]]
    feedback: str = state.feedback_msgs[-1].content
    remain_rounds: int = state.remain_rounds

    if len(msg) == 0:
        init_gen_text: str = state.init_data[conf["init_gen_text_col"]]
        msg.append(SystemMessage(
            content=REFINE_SYS_PROMPT\
                .replace("__SRC_DOC__", src_doc)\
                .replace("__INSTRUCTION__", instruction)
        ))
        msg.append(AIMessage(content=init_gen_text))
    msg.append(HumanMessage(content=feedback))

    if "no problem found" in feedback.lower():
        return {"refine_msgs": msg, "remain_rounds": remain_rounds}
    
    resp: AIMessage = llm.invoke(msg)
    msg.append(resp)
    remain_rounds -= 1
    return {"refine_msgs": msg, "remain_rounds": remain_rounds}


async def feedback(state: State, config: Config) -> Dict:
    llm: BaseChatModel = state.llm
    conf: Dict = config["configurable"]
    msg: List[AnyMessage] = state.feedback_msgs
    src_doc: str = state.init_data[conf["src_text_col"]]
    init_gen_text: str = state.init_data[conf["init_gen_text_col"]] 
    instruction: str = state.init_data[conf["instruction_col"]]

    if len(msg) == 0:
        msg.append(SystemMessage(
            content=FEEDBACK_SYS_PROMPT\
                .replace("__SRC_DOC__", src_doc)\
                .replace("__INSTRUCTION__", instruction)
        ))
        msg.append(HumanMessage(content=init_gen_text))
    else:
        msg.append(
            HumanMessage(content=state.refine_msgs[-1].content)
        )
    resp: AIMessage = llm.invoke(msg)
    msg.append(resp)
    return {"feedback_msgs": msg}


def refine_route(state: State) -> Literal["feedback", END]:
    remain_rounds: int = state.remain_rounds 
    feedback_msgs: List[AnyMesssage] = state.feedback_msgs
    if (
        remain_rounds == 0 
        or "no problem found" in feedback_msgs[-1].content.lower()
    ):
        return END
    else:
        return "feedback"


def graph_build(
    feedback: Callable, 
    refine: Callable
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(State, config_schema=Config)
    builder.add_node("feedback", feedback)
    builder.add_node("refine", refine)
    builder.add_edge(START, "feedback")
    builder.add_edge("feedback", "refine")
    builder.add_conditional_edges("refine", refine_route)
    graph: CompiledStateGraph = builder.compile()
    return graph


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    in_data_path: str = configs["in_data_path"]
    out_data_path: str = configs["out_data_path"]
    max_rounds: int = configs["max_rounds"]
    
    langchain_init(configs["langchain"])
    llm: BaseChatModel = init_chat_model(**configs["llms"][0])
    config: Dict = {
        "configurable": Config(
          src_text_col=configs["src_text_col"],
          init_gen_text_col=configs["init_gen_text_col"],
          instruction_col=configs["instruction_col"]
        ).dict() 
    }
    graph: CompiledStateGraph = graph_build(feedback, refine)
    samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ][:configs["max_sample_size"]]
    file = open(out_data_path, "w")
    for sample in tqdm(samples):
        inputs: Dict = {
            "init_data": sample, "llm": llm, "remain_rounds": max_rounds
        }
        try:
            result: AddableValuesDict = await graph.ainvoke(inputs, config=config)
        except Exception as e:
            print(e)
            continue
        output: str = result["refine_msgs"][-2].content
        sample["refine_output"] = output
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    file.close()
    print("Results are dumped to '%s'" % out_data_path)
    return


if __name__ == "__main__":
    asyncio.run(main())
