"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any, Dict, Optional
from llama_index.core import VectorStoreIndex, PromptTemplate
from modules.llm_interface import create_watsonx_llm
import config

logger = logging.getLogger(__name__)

# Function 1: generate_initial_facts(index) -> creates engaging conversation starters based on a person's LinkedIn profile
def generate_initial_facts(index: VectorStoreIndex) -> str:
    try:
        watsonx_llm = create_watsonx_llm(temperature = 0.0,
                                         max_new_tokens = 500,
                                         decoding_method = "sample")
        
        facts_prompt = PromptTemplate(template = config.INITIAL_FACTS_TEMPLATE)

        query_engine = index.as_query_engine(streaming = False,
                                             similarity_top_k = config.SIMILARITY_TOP_K,
                                             llm = watsonx_llm,
                                             text_qa_template = facts_prompt)
        
        query = "Provide three interesting facts about this person\'s career or education."
        response = query_engine.query(query)
        return response.response
      
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."

# Function 2: answer_user_query(index, user_query) -> powers the interactive Q&A capability of the bot
def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    try:
        watsonx_llm = create_watsonx_llm(temperature = 0.0,
                                         max_new_tokens = 250,
                                         decoding_method = "greedy")

        question_prompt = PromptTemplate(template = config.USER_QUESTION_TEMPLATE)

        base_retriever = index.as_retriever(similarity_top_k = config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query)
        context_str = "\n\n".join([node.node.get_text() for node in source_nodes])
        
        query_engine = index.as_query_engine(streaming = False,
                                             similarity_top_k = config.SIMILARITY_TOP_K,
                                             llm = watsonx_llm,
                                             text_qa_template = question_prompt)

        answer = query_engine.query(user_query)
        return answer
      
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
