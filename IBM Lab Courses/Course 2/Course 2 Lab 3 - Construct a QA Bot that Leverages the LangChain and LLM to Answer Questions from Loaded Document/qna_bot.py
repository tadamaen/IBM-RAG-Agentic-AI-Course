# Import the necessary libraries
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

# Suppress Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Initialize The LLM
def get_llm():
    model_id = 'mistralai/mistral-medium-2505'
    parameters = {GenParams.MAX_NEW_TOKENS : 256,
                  GenParams.TEMPERATURE : 0.5}
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(model_id = model_id,
                             url = "https://us-south.ml.cloud.ibm.com",
                             project_id = project_id,
                             params = parameters)
    return watsonx_llm

# Function 1: Define The Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

# Function 2: Define The Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                                   chunk_overlap = 50,
                                                   length_function = len)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function 3: Define The Vector Store
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# Function 4: Define The Embedding Model
def watsonx_embedding():
    embed_params = {EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS : 3,
                    EmbedTextParamsMetaNames.RETURN_OPTIONS : {"input_text" : True}}
    watsonx_embedding = WatsonxEmbeddings(model_id = "ibm/slate-125m-english-rtrvr-v2",
                                          url = "https://us-south.ml.cloud.ibm.com",
                                          project_id = "skills-network",
                                          params = embed_params)
    return watsonx_embedding

# Function 5: Define The Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# Function 6: Define A Question-Answering Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm = llm, 
                                     chain_type = "stuff", 
                                     retriever = retriever_obj, 
                                     return_source_documents = False)
    response = qa.invoke(query)
    return response['result']

# Setting Up The Gradio Interface
rag_application = gr.Interface(fn = retriever_qa,
                               allow_flagging = "never",
                               inputs = [gr.File(label = "Upload PDF File", file_count = "single", file_types = ['.pdf'], type = "filepath"),
                                         gr.Textbox(label = "Input Query", lines = 5, placeholder = "Type your question here...")],
                               outputs = gr.Textbox(label = "Output"),
                               title = "RAG Chatbot",
                               description = "Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.")

# Launch The Chatbot QnA Interface
rag_application.launch(server_name = "0.0.0.0", server_port = 7860, share = True)
