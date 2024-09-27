# -*- coding: utf-8 -*-
# file: benchmark_llm_tabular_understanding.py
# date: 2024-09-26


import pdb
import sys
import os
import json
from tqdm import tqdm
from typing import List, Dict, Optional
from pandas import DataFrame
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


PROMPT_SYS_TEMP: str = \
"""
You are a clinical data analyst.
Your task is to answer questions based on given tabular. 

# Tabular Schema
__TAB_DESC__
""".strip("\n")


PROMPT_USER_TEMP: str = \
"""
# Tabular
__TAB__

# Questions to Answer
__QUESTIONS__

# Response Format
{
  "question 1": "answer 1", 
  "question 2": "answer 2",
  ...
}
`question` are `string`, which have to be exactly same with given questions.
`answer` are `string`.

You must only return the JSON without anything else. 
"""


def llm_resp_json_clean(llm_json: str) -> str:
    return llm_json\
        .replace("```json", "")\
        .replace("```", "")


def json2jsonl_tabular(
    json_obj: Dict, 
    target_cols: List[str]
) -> List[Dict[str, str]]:
    out: List[Dict] = []
    cnt: int = len(json_obj[target_cols[0]])
    for i in range(cnt):
        out_sample: Dict = {}
        for field in target_cols:
            out_sample[field] = str(json_obj[field][i])
        out.append(out_sample)
    return out


def json2csv_tabular(
    json_obj: Dict,
    target_cols: List[str]
) -> DataFrame:
    return DataFrame(
        json2jsonl_tabular(json_obj, target_cols)
    )


def json2jsonl_tabular_str(
    json_obj: Dict,
    target_cols: List[str]
) -> str:
    out: str = ""
    for sample in json2jsonl_tabular(json_obj, target_cols):
        out += json.dumps(sample, ensure_ascii=False)
        out += "\n"
    return out


def tabular_desc_gen(
    tabular_schema: List[Dict],
    md_level: int=2
) -> str:
    out: str = ""
    md_level_sign: str = "#" * md_level
    for field in tabular_schema:
        name: str = field["name"]
        desc: str = field["description"]
        knowledges: Optional[List[str]] = field["knowledges"]
        out += "{} {}\n".format(md_level_sign, name)
        out += desc
        out += "\n\n"
        if knowledges is not None and len(knowledges) > 0:
            out += "#" * (md_level + 1) + " References of {}".format(name)  
            for knowledge in knowledges:
                out += "\n"
                out += "* {}".format(knowledge)
            out += "\n\n"
    return out


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    input_data_path: str = configs["input_data_path"]
    output_data_path: str = configs["output_data_path"]
    queries: List[str] = configs["queries"]
    tabular_schema: List[Dict] = configs["tabular_schema"]
    tabular_cols: str = [x["name"] for x in tabular_schema]

    samples: List[Dict] = [
        json.loads(x) 
        for x in open(input_data_path, "r").read().split("\n") if x != ""
        ][:configs["max_sample_size"]]
    llm: OpenAI = OpenAI(
        base_url=configs["llm"]["api_url"],
        api_key=configs["llm"]["api_key"],
    )
    tabular_desc: str = tabular_desc_gen(tabular_schema, 2)
    init_msg: List[Dict] = [
        {
            "role": "system", 
            "content": PROMPT_SYS_TEMP\
                .replace("__TAB_DESC__", tabular_desc)
        }
    ]
    out_file = open(output_data_path, "w")
    for sample in tqdm(samples):
        df: DataFrame = json2csv_tabular(sample, tabular_cols)
        tabular_str: str = json2jsonl_tabular_str(sample, tabular_cols)
        user_msg = {
            "role": "user", 
            "content": PROMPT_USER_TEMP\
                .replace("__TAB__", tabular_str)\
                .replace("__QUESTIONS__", "* " + "\n* ".join(queries)) 
        }
        msgs: List[Dict] = init_msg + [user_msg]
        resp: ChatCompletion = llm.chat.completions.create(
            messages=msgs, model=configs["llm"]["model"],
            temperature=0.0, 
            top_p=0.1
        )
        try:
            out_json: str = llm_resp_json_clean(
                resp.choices[0].message.content
            )
            out: Dict[str, str] = json.loads(out_json)
            out_sample: Dict = {
                "jsonl_tabular": tabular_str, "results": out
            }
            out_file.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
        except Exception as e:
            print(e)
    out_file.close()
    print("Results are dumped to '%s'" % output_data_path)
    return 


if __name__ == "__main__":
    main()
