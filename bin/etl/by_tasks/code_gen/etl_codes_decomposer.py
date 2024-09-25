# -*- coding: utf-8 -*-
# file: etl_extract_funcs_from_codes.py
# date: 2024-09-09


import pdb
import sys
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI


EXTENSION2LANG: Dict = {
    "cpp": "C++",
    "cc": "C++",
    "py": "Python",
    "ts": "TypeScript",
    "js": "JavaScript",
    "java": "Java"
}


SYSTEM_PROMPT_TEMP_FOR_DECOMP_PROMPT_GEN: str = \
"""
You are a master prompt engineer.
Your task is to build a prompt template, with which you will prompt yourself to decompose code into implementation units.

## Definition of Implementation Units
One of:
* Function together with its dependencies.
* Member function contained in class or struct implementation, together with its dependencies.

## Decomposition Task Input
A string, which is a __LANG__ source code. 
In your prompt the input placeholder should be "__CODE__". So when using your prompt, you can just replace "__CODE__" with real code.

## Decomposition Task Output
Decomposed code units (functions or member functions appended with necessary dependencies).
The format must be a JSON list without ```json ...```, for example:
[
  ["code snippet 1", "code snippet 1 dependencies include import clause, other variables or functions"],  
  ["code snippet 2", "code snippet 2 dependencies include import clause, other variables or functions"], 
  ...
]

Here are the requirements of the prompt template you generate:
* Generate 10 different in-context-learning examples to show how to extract implementation units from source code.
* Give a concrete example to show what's the output format looks like. 
* Always only return the prompt template without anything else in markdown format. 
""".strip("\n")


def src_file_get_meta(path: str) -> Dict:
    out: Dict = {
        "is_test": False,
        "file_name": "",
        "path": path, 
        "lang": None, 
        "tests": []
    }
    file_name: str = path.split("/")[-1]
    ext: str = file_name.split(".")[-1]
    out["file_name"] = file_name
    if ext in EXTENSION2LANG:
        out["lang"] = EXTENSION2LANG[ext]
    if ext in {"ts"}:
        if ".spec." in file_name or ".test." in file_name:
            out["is_test"] = True
        if out["is_test"] == False:
            out["tests"].append(file_name.replace(".ts", ".spec.ts"))
            out["tests"].append(file_name.replace(".ts", ".test.ts"))
    return out


def configs_reset(configs: Dict) -> None:
    if configs["llm_client"]["api_key"] == "":
        configs["llm_client"]["api_key"] = os.environ.get(
            configs["llm_client"]["api_key_env_var"],
            ""
        )


def prompt_to_decomp_msgs(code: str, temp: str) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = [
        ("user", temp.replace("__CODE__", code))
    ]
    return messages


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    configs_reset(configs)
    codes_jsonl_path: str = configs["codes_jsonl_path"]

    llm_cli: AzureChatOpenAI = AzureChatOpenAI(
        azure_deployment=configs["llm_client"]["deployment"],
        azure_endpoint=configs["llm_client"]["api_endpoint"],
        api_version=configs["llm_client"]["api_version"],
        api_key=configs["llm_client"]["api_key"],
    )
    data: List[Dict] = [
        json.loads(x) 
        for x in open(codes_jsonl_path, "r").read().split("\n") if x != ""
    ]
    impl_codes: List[Dict] = [
        x for x in data 
        if src_file_get_meta(x["path"])["is_test"] == False
    ]
    test_codes: List[Dict] = [
        x for x in data 
        if src_file_get_meta(x["path"])["is_test"] == True
    ]
    test_codes_kv: Dict[str, Dict] = {
        x["repo"] + ":" + x["path"].split("/")[-1]: x for x in test_codes
    }
    prompt_temp_decomps: Dict[str, str] = {}
    out_file = open(configs["output_path"], "w")
    for sample in tqdm(impl_codes):
        code: str = sample["code"]
        repo: str = sample["repo"]
        path: str = sample["path"]
        meta: Dict = src_file_get_meta(path)
        lang: str = meta["lang"]
        tests: List[str] = [
            repo + ":" + x for x in meta["tests"]
        ]
        exist_tests: List[str] = [x for x in tests if x in test_codes_kv]
        if len(exist_tests) == 0:
            print("No test found: {}".format(tests))
            continue
        test: str = tests[0]
        prompt_temp_decomp: str = ""
        if lang in prompt_temp_decomps:
            prompt_temp_decomp = prompt_temp_decomps[lang]
        else:
            prompt_temp_decomp: str = llm_cli.invoke(
                SYSTEM_PROMPT_TEMP_FOR_DECOMP_PROMPT_GEN.replace("__LANG__", lang)
            ).content
        msgs_in: List[Tuple[str, str]] = \
            prompt_to_decomp_msgs(code, prompt_temp_decomp)
        msg_out: str = llm_cli.invoke(msgs_in)
        msg_out_content: str = \
            msg_out.content.replace("```json", "").replace("```", "")
        code_units: List[str] = []
        try:
            code_units = json.loads(msg_out_content)
            print("Successed parsing")
            if lang not in prompt_temp_decomps:
                prompt_temp_decomps[lang] = prompt_temp_decomp
        except Exception as e:
            print("Failed parsing")
            continue
        for unit in code_units:
            impl: str = unit[0]
            deps: str = unit[1]
            sample_out: Dict = {
                "implementation": impl, 
                "dependencies": deps,
                "path": sample["path"], 
                "repo": sample["repo"],
                "impl_lines": len(impl.split("\n"))
            }
            out_file.write(json.dumps(sample_out, ensure_ascii=False) + "\n")

    out_file.close()
    return


if __name__ == "__main__":
    main()
