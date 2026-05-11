"""Module for interfacing with IBM watsonx.ai LLMs."""

import logging
from typing import Dict, Any, Optional

from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

import config

logger = logging.getLogger(__name__)

# Function 1: create_watsonx_embedding() -> creates the embedding model that converts text into vector representations
def create_watsonx_embedding() -> WatsonxEmbeddings:
    watsonx_embedding = WatsonxEmbeddings(model_id = config.EMBEDDING_MODEL_ID,
                                          url = config.WATSONX_URL,
                                          project_id = config.WATSONX_PROJECT_ID,
                                          truncate_input_tokens = 3)
  
    logger.info(f"Created Watsonx Embedding model: {config.EMBEDDING_MODEL_ID}")
    return watsonx_embedding

# Function 2: create_watsonx_llm() -> creates the language model that generates responses to user queries
def create_watsonx_llm(temperature: float = config.TEMPERATURE, max_new_tokens: int = config.MAX_NEW_TOKENS, decoding_method: str = "sample") -> WatsonxLLM:
    additional_params = {"decoding_method" : decoding_method,
                         "min_new_tokens" : config.MIN_NEW_TOKENS,
                         "top_k" : config.TOP_K,
                         "top_p" : config.TOP_P}
    
    watsonx_llm = WatsonxLLM(model_id = config.LLM_MODEL_ID,
                             url = config.WATSONX_URL,
                             project_id = config.WATSONX_PROJECT_ID,
                             temperature = temperature,
                             max_new_tokens = max_new_tokens,
                             additional_params = additional_params)
  
    logger.info(f"Created Watsonx LLM model: {config.LLM_MODEL_ID}")
    return watsonx_llm

# [Optional] Function 3: change_llm_model() -> allows to dynamically switch between different language models at runtime
def change_llm_model(new_model_id: str) -> None:
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
