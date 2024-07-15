# -*- codings: utf-8 -*-
# file: finetune_hf_lms.py
# date: 2024-07-04

"""
# Usage:
```
python ./bin/tools/hf_models_finetuner/finetune_mlm.py ./bin/tools/hf_models_finetuner/finetune_mlm.json
```

# References:
* https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
* https://towardsdatascience.com/how-to-train-bert-for-masked-language-modeling-tasks-3ccce07c6fdc
"""


import pdb
import sys
import os
import logging
import json
import re
import wandb
import os
import evaluate
import torch
import pandas as pd
from pprint import pprint
from datasets import load_dataset
from logging import Logger
from typing import Dict, List, Optional, Final, Callable, Union, Any, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from accelerate import Accelerator, DistributedType
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    is_torch_xla_available,
    set_seed,
)
from trl import SFTTrainer


LOGGER: Logger = logging.getLogger(__name__)


class CustomizedTextCollator(DataCollatorForLanguageModeling):
    def __init__(self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        target_text_cols: List[str], 
        col_separator: str="\n",
        mlm: bool=True, 
        mlm_probability: float=0.15, 
        pad_to_multiple_of: Optional[Any]=None, 
        return_tensors: str="pt"
    ):
        super().__init__(
            tokenizer=tokenizer, 
            mlm=mlm, 
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of, 
            return_tensors=return_tensors
        )
        self.target_text_cols: List[str] = target_text_cols
        self.col_separator: str = col_separator
        self.max_length: int = max_length

    def torch_call(self, 
        examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Only accept JSON lines dataset
        assert(isinstance(examples[0], dict))
        for col in self.target_text_cols:
            assert(col in examples[0])

        model_inputs: List[Dict[str, Tensor]] = []
        for example in examples:
            texts: List[str] = [
                example[x] for x in self.target_text_cols
            ]
            text: str = self.col_separator.join(texts)
            if len(text) == 0:
                LOGGER.warning("One sample has empty string value")
            model_input: Dict[str, Tensor] = self.tokenizer(
                text, 
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True
            )
            model_inputs.append(model_input)
        return super().torch_call(examples=model_inputs)
            

def dataset_load(
    path_or_name: str, 
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


def preprocess_logits_for_metrics(
    logits: Union[Tensor, Tuple[Tensor, Any]], 
    labels: Tensor
) -> Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    metric = evaluate.load(
        "accuracy", 
        #cache_dir=""
    )
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    out = metric.compute(predictions=preds, references=labels)
    print(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    LOGGER.info("Configs:\n{}".format(configs))
    model_path_or_name: str = configs["model_path_or_name"]
    tokenizer_path_or_name: str = configs["tokenizer_path_or_name"]
    output_path: str = configs["output_path"]
    cache_path: str = os.path.join(output_path, "./_cache")
    target_text_col: str = configs["target_text_col"]
    max_seq_length: int = configs["max_seq_length"]
    device: str = configs["device"]
    if device != "cpu":
        gpu_id: str = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        #torch.cuda.set_device(int(gpu_id))
        LOGGER.info("Using GPU %s" % gpu_id)

    # Loading datasets
    LOGGER.info("Loading datasets")
    train_samples: Dataset = dataset_load(
        configs["train_data_path_or_name"], configs["train_split"]
    )
    val_samples: Dataset = dataset_load(
        configs["val_data_path_or_name"], configs["val_split"]
    )
    if configs["rm_potential_data_leak"]:
        rm_keys: Set[str] = set(
            [x[target_text_col] for x in train_samples]
        )
        before_test_size: int = len(val_samples)
        val_samples = Dataset.from_pandas(
            pd.DataFrame(
              data=[x for x in val_samples if x[target_text_col] not in rm_keys]
            )
        )
        LOGGER.info("Original val data set: {}".format(len(val_samples)))
    LOGGER.info(
        "Train data size: {}, val data size: {}".format(
            len(train_samples), len(val_samples)
        )
    )

    # Initialize model and tokenizer
    LOGGER.info("Initializing model and tokenizer")
    model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
        model_path_or_name,
        from_tf=False,
        config=AutoConfig.from_pretrained(model_path_or_name),
        cache_dir=cache_path,
        #revision=model_args.model_revision,
        #token=model_args.token,
        trust_remote_code=True,
        #torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name, 
        cache_dir=cache_path,
        use_fast=False,
        #token=model_args.token,
        trust_remote_code=True
    )

    # Setting some variables
    LOGGER.info("Seting some variables")
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    collator: CustomizedTextCollator = \
        CustomizedTextCollator(
            tokenizer=tokenizer,
            target_text_cols=[target_text_col],
            col_separator="\n\n",
            max_length=max_seq_length,
            mlm=True, 
            mlm_probability=0.15, 
            pad_to_multiple_of=None, 
            return_tensors="pt"
        )
    collator_demo: Dict[str, Tensor] = collator.torch_call(
        [train_samples[0], train_samples[1]]
    )
    LOGGER.info("Collator example output:\n{}".format(collator_demo))
    training_args: TrainingArguments = TrainingArguments(
        output_dir=output_path, 
        overwrite_output_dir=False, 
        num_train_epochs=configs["epochs"],
        per_device_train_batch_size=configs["per_device_train_batch_size"],
        save_steps=configs["save_steps"],
        logging_steps=configs["logging_steps"],
        hub_private_repo=False,
        save_safetensors=True,
        learning_rate=configs["learning_rate"],
        report_to=configs["report_to"],
        save_total_limit=4,
        use_cpu = (configs["device"] == "cpu"),
        remove_unused_columns=False
    )
    trainer: Trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_samples,
        eval_dataset=val_samples,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    # Training
    LOGGER.info("Training")
    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    metrics = train_result.metrics
    pdb.set_trace()
    return


if __name__ == "__main__":
    main()
