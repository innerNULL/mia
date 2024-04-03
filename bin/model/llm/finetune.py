# -*- coding: utf-8 -*-
# file: run_finetuning.py
# date: 2024-03-25

"""
I hope this can be a self-contained implementation besides some  
common used libs like torch, transformers, ... etc, following 
are the dependencies you need:

Reference:
* https://wandb.ai/mostafaibrahim17/ml-articles/reports/Fine-Tuning-LLaMa-2-for-Text-Summarization--Vmlldzo2NjA1OTAy
"""


import pdb
import sys
import os
import json
import re
import wandb
import os
import torch
import pandas as pd
from pprint import pprint
from datasets import load_dataset
from typing import Dict, List, Optional, Final, Callable, Union
from torch import Tensor
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from trl import SFTTrainer


def gen_prompt(
    template_lines: List[str],
    sys_prompt: str, input_text: str, target_text: Optional[str], 
    is_inference: bool=False, 
    system_prompt_placeholder: str="__SYSTEM_PROMPT__", 
    input_text_placeholder: str="__INPUT_TEXT__",
    target_text_placeholder: str="__TARGET_TEXT__"
) -> str:
    if isinstance(template_lines, str):
        template_lines = [template_lines]
    template: str = "\n".join(template_lines)
    
    assert(system_prompt_placeholder in template)
    assert(input_text_placeholder in template)

    if not is_inference:
        assert(target_text_placeholder in template)
    else:
        template = template.replace(target_text_placeholder, "")
    
    template = template.strip("\n")
    prompt: str = template\
        .replace(system_prompt_placeholder, sys_prompt)\
        .replace(input_text_placeholder, input_text)
    if not is_inference:
        prompt = prompt.replace(target_text_placeholder, target_text)

    return prompt


def dataset_load(
    path_or_name: str, input_text_col: str, target_text_col: str, 
    split: Optional[str]=None
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


def datasets_load(
    train_path_or_name: str, 
    dev_path_or_name: str,
    test_path_or_name: str, 
    input_text_col: str, target_text_col: str
) -> DatasetDict:
    datasets: DatasetDict = DatasetDict()
    datasets["train"] = dataset_load(
        train_path_or_name, input_text_col, target_text_col, "train"
    )
    datasets["validation"] = dataset_load(
        dev_path_or_name, input_text_col, target_text_col, "validation"
    )
    datasets["test"] = dataset_load(
        test_path_or_name, input_text_col, target_text_col, "test"
    )
    return datasets


def datasets_processor_generator(
    input_text_col: str, target_text_col: str,
    template_lines: List[str],
    sys_prompt: str, 
    system_prompt_placeholder: str="__SYSTEM_PROMPT__", 
    input_text_placeholder: str="__INPUT_TEXT__",
    target_text_placeholder: str="__TARGET_TEXT__"
) -> Callable:
    """
    In future if we really need a complicated data processing  
    logic, just replace this with a `Callable` class.
    """
    def _processor(sample: Dict) -> Dict:
        input_text: str = sample[input_text_col]
        target_text: str = sample[target_text_col]
        prompt: str = gen_prompt(
            template_lines, sys_prompt, input_text, target_text,
            is_inference=False,
            system_prompt_placeholder=system_prompt_placeholder, 
            input_text_placeholder=input_text_placeholder, 
            target_text_placeholder=target_text_placeholder
        )
        return {
            input_text_col: input_text, target_text_col: target_text, 
            "prompt": prompt
        }

    return _processor


def model_init(
    path_or_name: str, quantization: bool, token: Optional[str]=None
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if quantization else torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        #use_safetensors=False,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        token=token
    )
    return model


def model_inference_with_decoding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: Union[str, List[str]],
    device: str,
    max_new_tokens: int=512
) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(torch.device(device))
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.0001
        )
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


def tokenizer_init(path_or_name: str, token: Optional[str]=None):
    tokenizer = AutoTokenizer.from_pretrained(
        path_or_name, token=token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    common_configs: Dict = configs["common"]
    data_configs: Dict = configs["data"]
    model_configs: Dict = configs["model"]
    train_configs: Dict = configs["train"]

    print(common_configs)
    print(data_configs)
    print(model_configs)
    print(train_configs)

    datasets_processor: Callable = datasets_processor_generator(
        input_text_col=data_configs["input_text_col"],
        target_text_col=data_configs["target_text_col"],
        template_lines=train_configs["prompt_temp"],
        sys_prompt=train_configs["system_prompt"]
    )
    datasets: DatasetDict = datasets_load(
        train_path_or_name=data_configs["train_path_or_name"], 
        dev_path_or_name=data_configs["dev_path_or_name"], 
        test_path_or_name=data_configs["test_path_or_name"],
        input_text_col=data_configs["input_text_col"],
        target_text_col=data_configs["target_text_col"]
    )
    datasets = datasets.map(datasets_processor, {})

    print("Raw Sample: ")
    print(datasets["train"][0])
    print("Processed Sample:")
    print(datasets_processor(datasets["train"][0]))

    model: AutoModelForCausalLM = model_init(
        path_or_name=model_configs["path_or_name"], 
        quantization=model_configs["quantization"],
        token=common_configs["hf_token"]
    )
    tokenizer = tokenizer_init(
        model_configs["path_or_name"], common_configs["hf_token"]
    )
    print("Model Quantization Info:")
    print(model.config.quantization_config.to_dict())

   
    example_raw_sample: str = datasets["train"][0]
    example_prompt: str = datasets_processor(example_raw_sample) 
    inference_example = model_inference_with_decoding(
        model, tokenizer, datasets["train"][0]["prompt"], 
        train_configs["device"], train_configs["max_new_tokens"]
    )
    print("example raw sample")
    print(example_raw_sample)
    print("example sample with prompt")
    print(example_prompt)
    print("exampel model input")
    print(inference_example)
    
    lora_config: LoraConfig = LoraConfig(
        lora_alpha=train_configs["lora_alpha"],
        lora_dropout=train_configs["lora_dropout"],
        r=train_configs["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=train_configs["learning_rate"],
        fp16=model_configs["quantization"],
        max_grad_norm=0.3,
        num_train_epochs=train_configs["num_epochs"],
        evaluation_strategy="epoch",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=7,
        group_by_length=True,
        output_dir=train_configs["ckpt_dir"],
        report_to="wandb",  # Set report_to here
        save_safetensors=False,
        lr_scheduler_type="cosine",
        seed=42,
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        peft_config=lora_config,
        dataset_text_field="prompt",
        max_seq_length=train_configs["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()

    pdb.set_trace()
