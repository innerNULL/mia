# -*- coding: utf-8 -*-
# file: auto_structured_text_summ.py
# date: 2024-09-12
"""
## Set Python Env
```shell
conda create -p ./_auto_structured_prompt python=3.11
conda activate ./_auto_structured_prompt
```
or
```
python3.11 -m auto_structured_prompt ./_auto_structured_prompt --copies
source ./_venv/bin/activate

pip install -r ./bin/poc/auto_structured_prompt.txt
```

## Run Examples
```shell
python ./bin/poc/auto_structured_ie.py ./bin/poc/auto_structured_ie.progress_note.json
python ./bin/poc/auto_structured_ie.py ./bin/poc/auto_structured_ie.image_report.json
```
"""


import pdb
import sys
import os
import json
import random as rd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from datasets import disable_caching
from typing import Dict, List, Optional, Callable, Union
from datasets import Dataset, DatasetDict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI
from datasets import load_dataset
from datasets import disable_caching
from datasets import Dataset, DatasetDict


def dataset_load(
    path_or_name: str, split: Optional[str]=None
) -> List[Dict]:
    if os.path.exists(path_or_name):
        if path_or_name.split(".")[-1] == "csv":
            return Dataset.from_pandas(pd.read_csv(path_or_name))
        elif path_or_name.split(".")[-1] == "jsonl":
            return load_dataset("json", data_files=path_or_name)["train"]
        else:
            raise Exception("Not a supported file format")
    else:
        if split is None:
            raise "Can not loading HuggingFace dataset without split info"
        return [dict(x) for x in load_dataset(path_or_name, split=split)]


def llm_resp_json_clean(llm_json: str) -> str:
    return llm_json\
        .replace("```json", "")\
        .replace("```", "")


def field_build_schema_desc(schema: Dict[str, str]) -> str:
    temp: str = "{} ({}): {}"
    return temp.format(schema["field"], schema["type"], schema["description"])


def fields_build_schema_desc(schemas: List[Dict[str, str]]) -> str:
    return "* " + "\n* ".join(
        [field_build_schema_desc(x) for x in schemas]
    )


def llm_gen_output_format_icl_examples(
    llm: BaseLanguageModel, 
    schemas: List[Dict[str, str]],
    cnt: int=1,
    prompt: Optional[str]=None
) -> str:
    if prompt is None:
        prompt: str = (
            "With a given schema, please generate __CNT__ example data. "
            "Your output have to be a JSON array.\n"
            "\n"
            "# Schema\n"
            "__SCHEMAS__\n"
            "\n"
            "Must only return JSON array without any words else."
        )
    prompt = prompt\
        .replace("__CNT__", str(cnt))\
        .replace("__SCHEMAS__", fields_build_schema_desc(schemas))
    print("Output format ICL examples generation prompt:\n%s" % prompt)
    llm_resp: str = llm.invoke(prompt)
    cleaned_json: str = llm_resp_json_clean(llm_resp.content)
    try:
        return json.loads(cleaned_json)
    except Exception as e:
        print(llm_resp.content)
        raise e
    

def prompt_sys_build(role: str, input_desc: str, task: str) -> str:
    return " ".join([role, input_desc, task])


def prompt_build_temp(
    llm: BaseLanguageModel,
    temp: Union[str, List[str]],
    role: str, 
    input_desc: str, 
    task: str,
    output_schemas: List[Dict[str, str]],
    requirements: List[str]
) -> str:
    out_fmt_examples: List[Dict] =  llm_gen_output_format_icl_examples(
        llm, output_schemas, 5, None
    )
    out: str = temp if isinstance(temp, str) else "".join(temp)
    out = out.replace(
        "__SYS_PROMPT__", prompt_sys_build(role, input_desc, task)
    )
    out = out.replace(
        "__OUTPUT_FMT__", "\n\n".join([
            json.dumps(x, indent=2, ensure_ascii=False) for x in out_fmt_examples
        ])
    )
    out = out.replace(
        "__OUTPUT_SCHEMA__", fields_build_schema_desc(output_schemas)
    )
    out = out.replace(
        "__OUTPUT_REQ__", "* " + "\n* ".join(requirements)
    )
    print("Main prompt template:\n%s" % out)
    return out


class LlmInfoEctractionAgent:
    def __init__(self):
        self.llm: Optional[BaseLanguageModel] = None
        self.temp: Optional[str] = None
        self.role: Optional[str] = None
        self.input_desc: Optional[str] = None
        self.task: Optional[str] = None
        self.output_schemas: Optional[List[Dict[str, str]]] = None
        self.requirements: Optional[List[str]] = None
        self.prompt_temp: Optional[str] = None

    @classmethod
    def new(cls, 
        llm: BaseLanguageModel,
        temp: Union[str, List[str]],
        role: str,
        input_desc: str,
        task: str,
        output_schemas: List[Dict[str, str]],
        requirements: List[str]
    ):
        out = cls()
        out.llm = llm
        out.prompt_temp = prompt_build_temp(
            llm, 
            temp=temp,
            role=role,
            input_desc=input_desc,
            task=task,
            output_schemas=output_schemas,
            requirements=requirements
        )
        return out

    def run(self, input_text: str) -> Dict:
        prompt: str = self.prompt_temp.replace("__DOC__", input_text)
        resp = self.llm.invoke(prompt)
        json_str: str = llm_resp_json_clean(resp.content)
        try:
            return json.loads(json_str)
        except Exception as e:
            print(resp.content)
            print(json_str)
            raise e


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    input_text_col: str = configs["input_text_col"]
    target_text_col: str = configs["target_text_col"]
    out_input_text_col: str = configs["out_input_text_col"]
    out_target_text_col: str = configs["out_target_text_col"]
    prompt_configs: Dict = configs["prompt"]
    output_schema: List[Dict] = prompt_configs["output_schema"]
    
    samples: List[Dict] = [
        x for x in 
        dataset_load(configs["data_path_or_name"], configs["data_split"])
    ][:configs["max_sample_size"]]
    llm: Optional[Union[BaseLanguageModel]] = ChatOpenAI(
        model=configs["llm"],
        api_key=configs["llm_api_key"],
        base_url=configs["llm_engine_api"],
        temperature=configs["temperature"],
        top_p=0.1
    )
    ie_agent: LlmInfoEctractionAgent = LlmInfoEctractionAgent.new(
        llm, 
        temp=prompt_configs["template"],
        role=prompt_configs["role_description"],
        input_desc=prompt_configs["input_description"],
        task=prompt_configs["task_description"],
        output_schemas=output_schema,
        requirements=prompt_configs["requirements"]
    )
    out_file = open(configs["output_path"], "w")
    for sample in tqdm(samples):
        try:
            input_text: str = sample[input_text_col]
            target_text: Optional[str] = sample.get(target_text_col, None)
            out: Dict = ie_agent.run(input_text)
            out_sample: Dict = {
                out_input_text_col: input_text, out_target_text_col: target_text
            }
            out_sample["ie_results"] = out
            out_file.write(json.dumps(out_sample,  ensure_ascii=False) + "\n")
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as e:
            print(e)
    out_file.close()
    print("Inference results are dumped to '%s'" % configs["output_path"])
    return


if __name__ == "__main__":
    main()
