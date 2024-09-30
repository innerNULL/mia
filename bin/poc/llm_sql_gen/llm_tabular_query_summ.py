# -*- coding: utf-8 -*-
# file: llm_tabular_abnormality_detection.py
# date: 2024-09-27
"""
## LLM Server Deployment
### vLLM
```shell
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.2-3B-Instruct --dtype bfloat16 --port 8081
```
## References
* https://arxiv.org/abs/2303.17651
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
You are a data engineer.
Your task is to write DuckDB SQL to generate table based on given input tables. 
Each column marks one specific abnormality. 

# Input Table Schema
__IN_SCHEMAS__

# Output Table Schema
__OUT_SCHEMA__

# Somethings to Note
__NOTES__
""".strip("\n")


PEOMPT_USER_SQL_GEN_TEMP: str = \
"""
# Input Tables
__IN_TABLES__

Write a DuckDB SQL to get requested output table:
* The data are saved into a DataFrame named same as input tables.
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
    top_p: float=0.05
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
            val: str =  str(json_obj[field][i])
            if val in {"null", "none", "None", "NULL"}:
                val = "NA"
            out_sample[field] = val
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


def table_schema_desc_gen(
    table_schema: List[Dict],
    md_level: int=2
) -> str:
    out: str = ""
    md_header_prefix: str = "#" * md_level
    for field in table_schema:
        name: str = field["name"]
        type_name: str = field["type"]
        desc: List[str] = field["descriptions"]
        out += "{} {} (Data Type: {})\n".format(
            md_header_prefix, name, type_name
        )
        out += " ".join(desc)
        out += "\n\n"
    return out.strip("\n")


def table_schemas_desc_gen(
    table_schemas: Dict[str, List[Dict]], 
    md_level: int=2
) -> str:
    out: str = ""
    md_header_prefix: str = "#" * md_level
    for table_name, table_schema in table_schemas.items():
        schema_desc_md_level: int = md_level + 1
        schema_desc: str = table_schema_desc_gen(
            table_schema, schema_desc_md_level
        )
        out += "{} Table '{}'".format(md_header_prefix, table_name)
        out += "\n"
        out += schema_desc
        out += "\n\n"
    return out


class AgentLlmSqlGen:
    def __init__(self): 
        self.llm: Optional[OpenAI] = None
        self.sys_prompt_temp: Optional[str] = None
        self.sql_gen_prompt_temp: Optional[str] = None
        self.sql_fix_prompt_temp: Optional[str] = None
        
        self.in_table_schemas: Optional[Dict[str, List[Dict]]] = None
        self.out_table_schema: Optional[List[Dict]] = None
        self.sth_to_note: Optional[List[str]] = None
        
        self.duckdb_conn = duckdb.connect()
        self.msgs: List[Dict[str, str]] = []
        self.usable_sql: Optional[str] = None

        self.cache_input_tables: Dict[str, DataFrame] = {}

    @classmethod
    def new(cls, 
        llm: OpenAI,
        in_table_schemas: Dict[str, List[Dict]],
        out_table_schema: List[Dict],  
        sth_to_note: List[str],
        sys_prompt_temp: str=PROMPT_SYS_TEMP,
        sql_gen_prompt_temp: str=PEOMPT_USER_SQL_GEN_TEMP,
        sql_fix_prompt_temp: str=PROMPT_USER_SQL_FIX_TEMP
    ):
        out = cls()

        out.llm = llm
        out.sys_prompt_temp = sys_prompt_temp
        out.sql_gen_prompt_temp = sql_gen_prompt_temp
        out.sql_fix_prompt_temp = sql_fix_prompt_temp
        
        out.in_table_schemas = in_table_schemas
        out.out_table_schema = out_table_schema
        out.sth_to_note = sth_to_note

        out.out_table_schema_desc: str = table_schema_desc_gen(
            out_table_schema, 2
        )
        out.in_table_schemas_desc: str = table_schemas_desc_gen(
            in_table_schemas, 2
        )
        out.sth_to_note_desc: str = "Nothing special to note."
        if len(sth_to_note) > 0:
            out.sth_to_note_desc = "* " + "\n* ".join(sth_to_note)

        sys_prompt: str = PROMPT_SYS_TEMP\
            .replace("__IN_SCHEMAS__", out.in_table_schemas_desc)\
            .replace("__OUT_SCHEMA__", out.out_table_schema_desc)\
            .replace("__NOTES__", out.sth_to_note_desc)
        out.msgs = [
            {"role": "system", "content": sys_prompt}
        ]
        
        return out

    def run_tables_register(self, 
        tabular_jsons: Dict[str, Dict[str, List]]
    ) -> str:
        in_table_strs: str = ""
        for table_name, table_json in tabular_jsons.items():
            table_cols: List[str] = [
                x["name"] for x in self.in_table_schemas[table_name]
            ]
            df: DataFrame = json2csv_tabular(table_json, table_cols)
            self.duckdb_conn.register(table_name, df)
            table_str: str = json2csv_tabular_str(table_json, table_cols)
            in_table_strs += "## Table '{}'".format(table_name)
            in_table_strs += "\n"
            in_table_strs += table_str
            in_table_strs += "\n\n"
        in_table_strs = in_table_strs\
            .strip("\n")#.replace("None", "NULL")
        return in_table_strs

    def run_init_sql_gen(self, 
        tabular_jsons: Dict[str, Dict[str, List]]
  ) -> str:
        """
        tabular_json: The format should be like:
            {
              "col1": [...],
              "col2": [...],
              ...
            }
        """
        in_table_strs: str = self.run_tables_register(tabular_jsons)
        user_msg_sql_gen: Dict = {
            "role": "user", 
            "content": self.sql_gen_prompt_temp\
                .replace("__IN_TABLES__", in_table_strs)
        }
        self.msgs += [user_msg_sql_gen]
        resp: str = llm_openai_call(self.llm, self.msgs)
        cleaned_resp: str = llm_resp_code_clean(resp, "sql") 
        self.msgs += [
            {"role": "assistant", "content": cleaned_resp}
        ]
        return cleaned_resp.strip("\n")
    
    def run_sql(self, sql: str) -> Tuple[Optional[DataFrame], str]:
        out_df: Optional[DataFrame] = None
        msg: str = "success"
        try:
            out_df = self.duckdb_conn.query(sql).df()
        except duckdb.Error as e:
            msg = str(e) 
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
        return cleaned_resp.strip("\n")

    def run_sql_gen(self,
        tabular_jsons: Dict[str, Dict[str, List]],
        max_rounds: int=20
    ) -> str:
        if self.usable_sql is not None:
            return self.usable_sql

        gen_sql: str = self.run_init_sql_gen(tabular_jsons)
        out_df: Optional[DataFrame] = None
        msg: str = ""
        rounds: int = 0
        out_df, msg = self.run_sql(gen_sql)
        while msg != "success":
            print("==== SQL Self-Refine Round %i fix ====" % rounds)
            gen_sql = self.fix_sql(msg)
            out_df, msg = self.run_sql(gen_sql)
            print("* Err Message:\n%s" % msg)
            print("\n* Generated SQL:\n%s" % gen_sql)
            print("==== SQL Self-Refine Round %i end ====" % rounds)
            rounds += 1
        self.usable_sql = gen_sql
        return self.usable_sql

    def run(self,
        tabular_jsons: Dict[str, Dict[str, List]],
        max_rounds: int=20
    ) -> DataFrame:
        gen_sql: str = self.run_sql_gen(tabular_jsons, max_rounds)
        return self.run_sql(gen_sql)[0]


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    input_data_path: str = configs["input_data_path"]
    output_data_path: str = configs["output_data_path"]
    in_table_schemas: Dict[str, List[Dict]] = configs["in_table_schemas"]
    out_table_schema: List[Dict] = configs["out_table_schema"]
    sth_to_note: List[str] = configs["sth_to_note"]

    samples: List[Dict] = [
        json.loads(x) 
        for x in open(input_data_path, "r").read().split("\n") if x != ""
    ][:configs["max_sample_size"]]
    llm: OpenAI = OpenAI(
        base_url=configs["llm"]["api_url"],
        api_key=configs["llm"]["api_key"],
    )
    agent_sql_gen: AgentLlmSqlGen = AgentLlmSqlGen.new(
        llm=llm,
        in_table_schemas=in_table_schemas,
        out_table_schema=out_table_schema,
        sth_to_note=sth_to_note
    ) 
    
    out_file = open(output_data_path, "w")
    for i, sample in enumerate(tqdm(samples)):
        # Tmp solution
        out_df: DataFrame = agent_sql_gen.run(
            {"vital_signs": sample}
        )
        if i <= 0:
            print(agent_sql_gen.usable_sql)
        print(out_df)
    out_file.close()
    print("Results are dumped to '%s'" % output_data_path)
    return 


if __name__ == "__main__":
    main()
