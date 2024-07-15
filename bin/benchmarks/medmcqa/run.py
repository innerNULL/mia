# -*- coding: utf-8 -*-
# file: run.py
# date: 2024-07-12


import pdb
import sys
import os
import json
import google
import time
import random as rd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from datasets import disable_caching
from typing import Dict, List, Optional, Callable, Set
from datasets import Dataset, DatasetDict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from typing import Union, Optional, List, Any, Dict, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


DARASET_NAME: str = "openlifescienceai/medmcqa"


SUPPORTED_LLMS: Set[str] = {
    "medlm-large", 
    "medlm-medium", 
    "gpt-4",
    "gpt-4o"
}


PROMPT: str = \
"""
## Question
{question}

## Options
0: {op0}
1: {op1}
2: {op2}
3: {op3}

To answer above question, which option is correct? Only return option ID.
""".strip("\n")


class VertexAIMedLM(LLM):
    """Adapted  from https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/"""
    parameters_dict: dict = {
        "candidateCount": 1,
        "maxOutputTokens": 1024,
        "temperature": 0,
        "topP": 0.8,
        "topK": 40
    }
    model_name: str = "medlm-large"
    gcp_project_id: str

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.
        Override this method to implement the LLM logic.
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.
        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        client_obj = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        parameters = json_format.ParseDict(self.parameters_dict, Value())

        instance_dict = {
            "content": prompt
        }
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]

        response =  client_obj.predict(
            endpoint=f"projects/{self.gcp_project_id}/locations/us-central1/publishers/google/models/{self.model_name}",
            instances=instances, parameters=parameters
        )
        predictions = response.predictions
        if predictions and len(predictions) > 0:
            return predictions[0]['content']
        return "No predictions found"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "VertexAI-MedLM-Wrapper"


def init_llm_client(
    llm_engine_type: str, 
    llm_engine_api: Optional[str],
    llm: str, 
    sampling: bool=True, 
    temperature: float=0.01, 
    top_k: int=1,
    local_model_path: str="",
    llm_engine_version: str="",
    llm_api_key: str=""
) -> Optional[Union[BaseLanguageModel]]:
    if "medlm" in llm.lower():
        print("Using self-implemented `VertexAIMedLM`")
        credentials, project_id = google.auth.default()
        return VertexAIMedLM(model_name=llm, gcp_project_id=project_id)
    elif llm_engine_type in {"ollama", "Ollama"}:
        return ChatOllama(
            model=llm,
            base_url=llm_engine_api,
            temperature=temperature, 
            top_k=top_k
        )
    elif llm_engine_type in {"hf", "huggingface", "HuggingFace"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        llm_runner = HuggingFacePipeline.from_model_id(
            model_id=llm, 
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512, 
                "do_sample": sampling, 
                "temperature": temperature, 
                "top_k": top_k,
                "repetition_penalty": 1.03,
                "return_full_text": False
            },
            model_kwargs={
                "quantization_config": quant_config
            }
        )
        return ChatHuggingFace(llm=llm_runner)
    elif llm_engine_type in {"llama.cpp", "llamacpp"}:
        return ChatLlamaCpp(
            temperature=temperature,
            model_path=local_model_path,
            n_ctx=10000,
            n_gpu_layers=8,
            n_batch=300,
            max_tokens=512,
            n_threads=multiprocessing.cpu_count() - 1,
            repeat_penalty=1.03,
            #top_p=0.5,
            verbose=True,
        )
    elif llm_engine_type in {"vllm"}:
        return ChatOpenAI(
            model=llm,
            openai_api_key="EMPTY",
            openai_api_base="{}/v1".format(llm_engine_api),
            temperature=temperature,
            top_p=0.9
        )
    elif llm_engine_type in {"azure_chat_openai"}:
        print("Using `AzureChatOpenAI`")
        return AzureChatOpenAI(
            seed=42,
            temperature=temperature, 
            #top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            azure_deployment=llm, 
            model=llm,
            model_name=llm,
            api_key=llm_api_key,
            api_version=llm_engine_version,
            azure_endpoint=llm_engine_api
        )
    else:
        return None


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    max_data_size: int = configs["max_data_size"]
    output_path: str = configs["output_path"]
    
    samples: Dataset = load_dataset(DARASET_NAME, split="validation")
    llm: Optional[Union[BaseLanguageModel]] = init_llm_client(
        llm_engine_type=configs["llm_engine_type"],
        llm_engine_api=configs["llm_engine_api"],
        llm=configs["llm"],
        llm_engine_version=configs["llm_engine_version"],
        llm_api_key=configs["llm_api_key"],
    )
    chain: RunnableSequence = \
        ChatPromptTemplate.from_messages([("human", PROMPT)]) \
        | llm \
        | StrOutputParser()
    
    out_file = open(output_path, "w")
    succ_cnt: int = 0
    cnt: int = 0
    for sample in tqdm(samples):
        question: str = sample["question"]
        options: Dict[int, str] = {
            0: sample["opa"],
            1: sample["opb"],
            2: sample["opc"],
            3: sample["opd"]
        }
        option_correct: int = sample["cop"] 
        option_out: int = -1
        try:
            resp_text: str = chain.invoke(
                {
                    "question": question, 
                    "op0": options[0],
                    "op1": options[1],
                    "op2": options[2],
                    "op3": options[3]
                }
            )
            option_out = int(resp_text)
        except Exception as e:
            print(e)
        
        sample["output"] = option_out
        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        succ_cnt += int(option_out == option_correct)
        cnt += 1
        print("option_out={}, option_correct={}".format(option_out, option_correct))
        print("correct ratio: {}".format(succ_cnt / cnt))
        time.sleep(2)
        if cnt >= max_data_size:
            break
    
    out_file.close()
    print("Benchmarking results are dumped to: %s" % output_path)
    return


if __name__ == "__main__":
    main()
