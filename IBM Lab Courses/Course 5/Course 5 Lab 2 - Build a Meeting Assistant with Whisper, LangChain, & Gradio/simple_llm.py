from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  
from ibm_watsonx_ai import APIClient, Credentials  
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes 
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  

# Running a simple LLM
parameters = {GenParams.DECODING_METHOD : "sample",
              GenParams.MAX_NEW_TOKENS : 512,
              GenParams.MIN_NEW_TOKENS : 1,
              GenParams.TEMPERATURE : 0.5,
              GenParams.TOP_K : 50,
              GenParams.TOP_P : 1}
model_id = 'ibm/granite-3-3-8b-instruct'
project_id = "skills-network"

granite_llm = WatsonxLLM(model_id = model_id,
                         url = "https://us-south.ml.cloud.ibm.com",
                         project_id = project_id,
                         params = parameters)

response = granite_llm.invoke("How to read a book effectively?")
print(response)
