# -*- coding: utf-8 -*-
# file: inference.py
# date: 2024-04-13
#
# Usage:
# python ./bin/model/llm/clinical_note_generation_from_doctor_patient_conversations/inference.py demo_configs/model/llm/clinical_note_generation_from_doctor_patient_conversations/inference-azure-openai.json 


import pdb
import sys
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from datasets import disable_caching
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.ir import StructuredQuery


class LangchainEmbedding(Embeddings):
    """
    https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/embeddings/embeddings.py
    """
    def __init__(self):
        self.encoder: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device: torch.device = torch.device("cpu")

    def init(self,
        encoder: Union[str, PreTrainedModel], 
        tokenizer: Union[str, PreTrainedTokenizer], 
        device: str="cpu"
    ) -> None:
        self.encoder = \
            AutoModel.from_pretrained(encoder) if isinstance(encoder, str) \
            else encoder
        self.tokenizer = \
            AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) \
            else tokenizer
        self.device = torch.device(device)

        self.encoder = self.encoder.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in tqdm(texts):
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return text_embedding(
            text, self.encoder, self.tokenizer, False, self.device
        )


def text_embedding(
    text: str,
    encoder: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
    use_cls_embedding: bool=False,
    device: Union[str, torch.device]="cpu"
) -> List[float]:
    device = torch.device(device) if isinstance(device, str) else device
    tokens: BatchEncoding = tokenizer.encode_plus(
        text, add_special_tokens=False, max_length=512, truncation=True, 
        return_tensors='pt'
    ).to(torch.device(device))
    encoder = encoder.to(torch.device(device))
    
    encoder.eval()
    with torch.no_grad():
        embd: Tensor = None
        if use_cls_embedding:
            embd = encoder(**tokens)["last_hidden_state"][0, 0, :]
        else:
            embd = torch.mean(
                encoder(**tokens)["last_hidden_state"][:, 1:, :], dim=1
            ).squeeze()
    return embd.tolist()


def dataset_load(
    path_or_name: str, split: Optional[str]=None
) -> List[Dict]:
    out_dataset: Optional[Dataset] = None
    if os.path.exists(path_or_name):
        if path_or_name.split(".")[-1] == "csv":
            out_dataset = Dataset.from_pandas(pd.read_csv(path_or_name))
        elif path_or_name.split(".")[-1] == "jsonl":
            out_dataset = load_dataset("json", data_files=path_or_name)["train"]
        else:
            raise Exception("Not a supported file format")
    else:
        if split is None:
            raise "Can not loading HuggingFace datas"
        out_dataset = load_dataset(path_or_name, split=split)
    return [x for x in out_dataset]


def build_prompt(
    user_prompt: str, 
    icl_examples: List[str], 
    system_prompts: List[str], 
    icl_example_prefix: str, 
    user_prompt_template: List[str]
) -> str:
    system_prompt: str = "".join(system_prompts)
    user_prompt: str = "".join(user_prompt_template)\
        .replace("__CONVERSATION__", user_prompt)
    icl_examples_with_prefix: List[str] = []

    for i, icl_example in enumerate(icl_examples):
        icl_example_with_prefix: str = icl_example_prefix + " " + str(i) + ":" 
        icl_examples_with_prefix.append(icl_example_with_prefix + "\n" + icl_example)

    icl_prompt: str = "".join(icl_examples_with_prefix)

    return "\n".join([system_prompt, icl_prompt, user_prompt])



if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read()) 
    print(configs)
    #os.environ["AZURE_OPENAI_API_KEY"] = configs["llm_api_key"]
    #os.environ["AZURE_OPENAI_ENDPOINT"] = configs["llm_api_endpoint"]

    disable_caching()
    
    vec_engine: Optional[Union[FAISS]] = None
    llm: AzureChatOpenAI = AzureChatOpenAI(
        azure_deployment=configs["llm_deployment"],
        azure_endpoint=configs["llm_api_endpoint"],
        openai_api_version=configs["llm_api_version"],
        api_key=configs["llm_api_key"],
    )
    icl_samples: List[Dict] = dataset_load(
        configs["in_context_example_path_or_name"], 
        configs["in_context_example_split"]
    ) 
    inf_samples: List[Dict] = dataset_load(
        configs["inference_data_path_or_name"], 
        configs["inference_data_split"]
    )
    hf_encoder: PreTrainedModel = AutoModel.from_pretrained(
        configs["hf_encoder_path_or_name"]
    )
    hf_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs["hf_tokenizer_path_or_name"]
    )
    embd: LangchainEmbedding = LangchainEmbedding()
    embd.init(hf_encoder, hf_tokenizer, configs["device"])

    icl_examples: List[Document] = [
        Document(
            page_content=x[configs["soap_notes_col_name"]],
            metadata={}
        ) for x in icl_samples
    ]

    vec_engine = FAISS.from_documents(icl_examples, embd)
    retriever = vec_engine.as_retriever()
    
    out_file = open(configs["output_path"], "w")
    for sample in tqdm(inf_samples):
        conversation: str = sample[configs["conversation_col_name"]]
        cands: List[Document] = retriever.get_relevant_documents(
            conversation, k=configs["top_k"]
        )
        prompt: str = build_prompt(
            conversation, 
            [x.page_content for x in icl_examples][:configs["top_k"]],
            configs["system_prompt"], 
            configs["icl_example_prefix"], 
            configs["user_prompt_template"]
        )
        output: str = llm.invoke(prompt).content
        out_file.write(json.dumps(
            {"output_text": output, "prompt": prompt}, ensure_ascii=False
        ))
    out_file.close()
    print("Dumped inference result at: %s" % configs["output_path"])
