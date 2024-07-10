# -*- coding: utf-8 -*-
# file: lib.py
# date: 2024-06-29


import multiprocessing
import google.auth
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from typing import Union, Optional, List, Any, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig


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
    local_model_path: str=""
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
    else:
        return None
    
