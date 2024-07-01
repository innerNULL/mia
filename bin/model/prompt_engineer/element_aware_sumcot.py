# -*- coding: utf-8 -*-
# file: run_element_aware_sumcot.py
# date: 2024-06-29


import pdb
import sys
import os
import json
import random as rd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from datasets import disable_caching
from typing import Dict, List, Optional, Callable
from datasets import Dataset, DatasetDict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

import lib


ELEMENT_EXTRACT_PROMPT: str = (
    "Document:\n"
    "{doc}\n"
    "\n"
    "__QUESTIONS__\n"
    "\n"
    "Please Answer the above questions: "
)


SUMMARIZATION_PROMPT: str = (
    "Document:\n"
    "{doc}\n"
    "\n"
    "Let's integrate the following information and summarize above document in less than __MAX_WORDS__ words:\n"
    "{elements}\n"
    "\n"
    "The output format should be a JSON \"{{\"summary\": YOUR SUMMARIZATION RESULT}}\", \n"
    "and do not output anything else except this JSON."
)


def element_extract_prompt_build(
    element_extract_questions: List[str],
    prompt_temp: str=ELEMENT_EXTRACT_PROMPT
) -> ChatPromptTemplate:
    prompt_temp = prompt_temp.replace(
        "__QUESTIONS__", "\n".join(element_extract_questions)
    )
    print("Elements extraction prompt template:\n%s" % prompt_temp)
    return ChatPromptTemplate.from_messages([("system", prompt_temp)])


def summarization_prompt_build(
    maximum_output_words: int,
    prompt_temp: str=SUMMARIZATION_PROMPT
) -> ChatPromptTemplate:
    prompt_temp = prompt_temp.replace(
        "__MAX_WORDS__", str(maximum_output_words)
    )
    print("Summarization prompt template:\n%s" % prompt_temp)
    return ChatPromptTemplate.from_messages([("system", prompt_temp)])


def dataset_load(
    path_or_name: str, split: Optional[str]=None
) -> Dataset:
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
        return load_dataset(path_or_name, split=split)


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    input_text_col: str = configs["input_text_col"]
    target_text_col: str = configs["target_text_col"]
    maximum_output_words: int = configs["maximum_output_words"]
    
    samples: List[Dict] = [
        dict(x) for x in 
        dataset_load(configs["data_path_or_name"], configs["data_split"])
    ]
    llm: Optional[Union[BaseLanguageModel]] = None
    
    llm = lib.init_llm_client(
        llm_engine_type=configs["llm_engine_type"],
        llm_engine_api=configs["llm_engine_api"],
        llm=configs["llm"]
    )
    element_extract_prompt: ChatPromptTemplate = element_extract_prompt_build(
        element_extract_questions=configs["element_queries"],
        prompt_temp=ELEMENT_EXTRACT_PROMPT
    )
    element_extract_chain: RunnableSequence = \
        element_extract_prompt | llm | StrOutputParser()

    summarization_prompt: ChatPromptTemplate = summarization_prompt_build(
        maximum_output_words=maximum_output_words,
        prompt_temp=SUMMARIZATION_PROMPT
    )
    summarization_chain: RunnableSequence = \
        summarization_prompt | llm | StrOutputParser()

    output_file = open(configs["output_path"], "w")
    for sample in tqdm(samples):
        input_text: str = sample[input_text_col]
        target_text: str = sample[target_text_col]
        extracted_elements: str = element_extract_chain.invoke({"doc": input_text})
        summary: str = summarization_chain.invoke(
            {"doc": input_text, "elements": extracted_elements}
        ).replace("`", "").replace("json", "").strip("\n")
        output_text: str = ""
        try:
            output_text = json.loads(summary)["summary"]
        except Exception as e:
            print(e)
            print(summary)
        output: Dict = {
            input_text_col: input_text, target_text_col: target_text, 
            "output_text": output_text
        }
        output_file.write(json.dumps(output, ensure_ascii=False) + "\n")
        if configs["dbg_mode"]:
            print("output_text: %s" % output_text)
    
    output_file.close()
    print("Inference results are dumped to: %s" % configs["output_path"])
    return


if __name__ == "__main__":
    main()
