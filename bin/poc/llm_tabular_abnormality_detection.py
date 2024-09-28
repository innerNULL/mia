# -*- coding: utf-8 -*-
# file: llm_tabular_abnormality_detection.py
# date: 2024-09-27
"""
## References
* https://arxiv.org/abs/2308.03188
"""


import pdb
import sys
import os
import traceback
import json
import duckdb
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any, Union
from io import StringIO
from pandas import DataFrame
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


PROMPT_SYS_TEMP: str = \
"""
You are a clinical data analyst.
Your task is to write DuckDB SQL to extend original table with new boolean columns. 
Each column marks one specific abnormality. 

# Tabular Schema
__TAB_DESC__

# Abnormalities to Detect
__ABNORMALITIES__

# Notes
__NOTES__
""".strip("\n")


PEOMPT_USER_SQL_GEN_TEMP: str = \
"""
# Tabular
__TAB__

Write a DuckDB SQL to detect target abnormalities:
* The data are saved into a DataFrame nameed as `df`.
* Only return your SQL code without anything else.
""".strip("\n")


PROMPT_USER_SQL_FIX_TEMP: str = \
"""
You got following error:

__ERR__

Please fix it and only return your fixed SQL without anything else.
""".strip("\n")


def llm_openai_call(
    llm: OpenAI, 
    msgs: List[Dict],
    model: str="llama3.1:8b",
    temperature: float=0.0,
    top_p: float=0.1
) -> str:
    resp: ChatCompletion = llm.chat.completions.create(
        messages=msgs, 
        model=model,
        temperature=temperature, 
        top_p=top_p
    )
    return resp.choices[0].message.content


def llm_resp_json_clean(llm_json: str) -> str:
    return llm_json\
        .replace("```json", "")\
        .replace("```", "")


def llm_resp_code_clean(llm_code: str, lang: str="sql") -> str:
    return llm_code\
        .replace("```{}".format(lang), "")\
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


def json2csv_tabular_str(
    json_obj: Dict,
    target_cols: List[str]
) -> str:
    df: DataFrame = json2csv_tabular(json_obj, target_cols)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def tabular_desc_gen(
    tabular_schema: List[Dict],
    md_level: int=2
) -> str:
    out: str = ""
    md_level_sign: str = "#" * md_level
    for field in tabular_schema:
        name: str = field["name"]
        desc: str = field["description"]
        out += "{} {}\n".format(md_level_sign, name)
        out += desc
        out += "\n\n"
    return out.strip("\n")


def abnormalities_desc_gen(
    abnormalities: List[Dict],
    md_level: int=2
) -> str:
    out: str = ""
    md_level_sign: str = "#" * md_level
    for abnormality in abnormalities:
        topic: str = abnormality["topic"]
        points: str = "* " + "\n* ".join(abnormality["descriptions"])
        out += "{} {}".format(md_level_sign, topic)
        out += "\n"
        out += points
        out += "\n\n"
    return out.strip("\n")


class AgentLlmSqlGen:
    def __init__(self): 
        self.llm: Optional[OpenAI] = None
        self.tabular_cols: List[str] = []
        self.sys_prompt_temp: Optional[str] = None
        self.sql_gen_prompt_temp: Optional[str] = None
        self.sql_fix_prompt_temp: Optional[str] = None

        self.msgs: List[Dict[str, str]] = []
        self.df: Optional[DataFrame] = None

        self.usable_sql: Optional[str] = None

    @classmethod
    def new(cls, 
        llm: OpenAI,
        tabular_cols: List[str],
        tabular_desc: str, 
        abnorm_desc: str, 
        notes: str,
        sys_prompt_temp: str=PROMPT_SYS_TEMP,
        sql_gen_prompt_temp: str=PEOMPT_USER_SQL_GEN_TEMP,
        sql_fix_prompt_temp: str=PROMPT_USER_SQL_FIX_TEMP
    ):
        out = cls()

        out.llm = llm
        out.tabular_cols = tabular_cols
        out.sys_prompt_temp = sys_prompt_temp
        out.sql_gen_prompt_temp = sql_gen_prompt_temp
        out.sql_fix_prompt_temp = sql_fix_prompt_temp
        
        sys_prompt: str = PROMPT_SYS_TEMP\
            .replace("__TAB_DESC__", tabular_desc)\
            .replace("__ABNORMALITIES__", abnorm_desc)\
            .replace("__NOTES__", notes)
        out.msgs = [
            {"role": "system", "content": sys_prompt}
        ]
        
        return out
   
    def reset(self) -> None:
        self.df = None
        self.msgs = self.msgs[:1]

    def run_init_sql_gen(self, 
        tabular_json: Dict[str, List[Union[str, int, float]]]
  ) -> str:
        """
        tabular_json: The format should be like:
            {
              "col1": [...],
              "col2": [...],
              ...
            }
        """
        self.df: DataFrame = json2csv_tabular(
            tabular_json, self.tabular_cols
        )
        tabular_str: str = json2csv_tabular_str(
            tabular_json, self.tabular_cols
        )
        user_msg_sql_gen: Dict = {
            "role": "user", 
            "content": self.sql_gen_prompt_temp\
                .replace("__TAB__", tabular_str)
        }
        self.msgs += [user_msg_sql_gen]
        resp: str = llm_openai_call(self.llm, self.msgs)
        cleaned_resp: str = llm_resp_code_clean(resp, "sql") 
        self.msgs += [
            {"role": "assistant", "content": cleaned_resp}
        ]
        return cleaned_resp
    
    def run_sql(self, df: DataFrame, sql: str) -> Tuple[Optional[DataFrame], str]:
        out_df: Optional[DataFrame] = None
        msg: str = "success"
        try:
            out_df = duckdb.query(sql).df()
        except Exception as e:
            msg = traceback.format_exc()
        return out_df, msg

    def fix_sql(self, err_msg: str) -> str:
        self.msgs.append(
            {
              "role": "user", 
              "content": self.sql_fix_prompt_temp.replace("__ERR__", err_msg)
            }
        )
        resp: str = llm_openai_call(self.llm, self.msgs)
        cleaned_resp: str = llm_resp_code_clean(resp, "sql")
        self.msgs += [
            {"role": "assistant", "content": cleaned_resp}
        ]
        return cleaned_resp

    def run_sql_gen(self, 
        tabular_json: Dict[str, List[Union[str, int, float]]],
        max_rounds: int=20
    ) -> str:
        if self.usable_sql is not None:
            return self.usable_sql

        gen_sql: str = self.run_init_sql_gen(tabular_json)
        out_df: Optional[DataFrame] = None
        msg: str = ""
        rounds: int = 0
        out_df, msg = self.run_sql(self.df, gen_sql)
        while msg != "success":
            print("Round %i fix" % rounds)
            gen_sql = self.fix_sql(msg)
            out_df, msg = self.run_sql(self.df, gen_sql)
            rounds += 1
        self.usable_sql = gen_sql
        return self.usable_sql

    def run(self, 
        tabular_json: Dict[str, List[Union[str, int, float]]],
        max_rounds: int=20
    ) -> DataFrame:
        gen_sql: str = self.run_sql_gen(tabular_json, max_rounds)
        df: DataFrame = json2csv_tabular(
            tabular_json, self.tabular_cols
        )
        return self.run_sql(df, gen_sql)[0]


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    input_data_path: str = configs["input_data_path"]
    output_data_path: str = configs["output_data_path"]
    tabular_schema: List[Dict] = configs["tabular_schema"]
    tabular_cols: str = [x["name"] for x in tabular_schema]
    abnormalities: List[Dict] = configs["target_abnormalities"]

    samples: List[Dict] = [
        json.loads(x) 
        for x in open(input_data_path, "r").read().split("\n") if x != ""
    ][:configs["max_sample_size"]]
    llm: OpenAI = OpenAI(
        base_url=configs["llm"]["api_url"],
        api_key=configs["llm"]["api_key"],
    )

    tabular_desc: str = tabular_desc_gen(tabular_schema, 2)
    abnorm_desc: str = abnormalities_desc_gen(abnormalities, 2)
    notes_desc: str = "* " + "\n* ".join(configs["global_rules"])
    agent_sql_gen: AgentLlmSqlGen = AgentLlmSqlGen.new(
        llm=llm,
        tabular_cols=tabular_cols,
        tabular_desc=tabular_desc, 
        abnorm_desc=abnorm_desc,
        notes=notes_desc
    ) 
    
    out_file = open(output_data_path, "w")
    for sample in tqdm(samples):
        out_df: str = agent_sql_gen.run(sample)
        print(out_df)
    out_file.close()
    print("Results are dumped to '%s'" % output_data_path)
    return 


if __name__ == "__main__":
    main()
