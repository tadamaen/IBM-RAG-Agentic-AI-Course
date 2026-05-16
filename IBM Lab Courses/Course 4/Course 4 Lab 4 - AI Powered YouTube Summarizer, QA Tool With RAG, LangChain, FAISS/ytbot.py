# TERMINAL COMMANDS
# pip install virtualenv 
# virtualenv my_env
# source my_env/bin/activate

# INSTALLING AND IMPORTING PACKAGES
# pip install youtube-transcript-api==1.2.1
# pip install faiss-cpu==1.8.0
# pip install langchain==0.2.6 | tail -n 1
# pip install langchain-community==0.2.6 | tail -n 1
# pip install ibm-watsonx-ai==1.0.10 | tail -n 1
# pip install langchain_ibm==0.1.8 | tail -n 1
# pip install gradio==4.44.1 | tail -n 1
# python3.11 -m pip uninstall -y huggingface_hub
# python3.11 -m pip install huggingface_hub==0.16.4
# pip install --upgrade gradio fastapi starlette jinja2

# Import necessary libraries for the YouTube bot
import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Function To Extract YouTube Transcripts
def get_video_id(url):    
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_id = get_video_id(url)
print(video_id)

# Function To Fetch Transcripts From YouTube
def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    transcript = ""
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break
    return transcript if transcript else None

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
transcript = get_transcript(url)
print(transcript)

# Function To Process The Transcript
def process(transcript):
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
    return txt

transcript = [{"text" : "We're no strangers to love.", "start" : 0.0, "duration" : 3.5},
              {"text" : "You know the rules and so do I.", "start" : 3.5, "duration" : 4.0},
              {"text" : "A full commitment's what I'm thinking of.", "start" : 7.5, "duration" : 4.0}]
formatted_transcript = process(transcript)
print(formatted_transcript)

# Function To Chunking The Transcript
def chunk_transcript(processed_transcript, chunk_size = 200, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

processed_transcript = """Text: We're no strangers to love. Start: 0.0
Text: You know the rules and so do I. Start: 3.5
Text: A full commitment's what I'm thinking of. Start: 7.5"""

chunks = chunk_transcript(processed_transcript)
print(chunks)

# Setting Up Watsonx Model
def setup_credentials():
    model_id = "ibm/granite-8b-code-instruct"
    credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com")
    client = APIClient(credentials)
    project_id = "skills-network"
    return model_id, credentials, client, project_id

def define_parameters():
    return {GenParams.DECODING_METHOD : DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS : 900}

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    return WatsonxLLM(model_id = model_id, url = credentials.get("url"), project_id = project_id, params = parameters)

# Function to create and return an instance of WatsonxEmbeddings with the specified configuration
def setup_embedding_model(credentials, project_id):
    return WatsonxEmbeddings(model_id = 'ibm/slate-30m-english-rtrvr-v2', url = credentials["url"], project_id = project_id)

# Function to create FAISS Index
def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

# Function to perform similarity search
def perform_similarity_search(faiss_index, query, k = 3):
    results = faiss_index.similarity_search(query, k = k)
    return results

# Function to create the summary prompt
def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = PromptTemplate(input_variables = ["transcript"], template = template)
    return prompt

# Function to create an LLMChain for generating summaries
def create_summary_chain(llm, prompt, verbose = True):
    return LLMChain(llm = llm, prompt = prompt, verbose = verbose)

# Function to retrieve relevant context from the FAISS index based on the user's query
def retrieve(query, faiss_index, k = 7):
    relevant_context = faiss_index.similarity_search(query, k = k)
    return relevant_context

# Function to create a PromptTemplate for question answering based on video content
from langchain import PromptTemplate
def create_qa_prompt_template():
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """
    prompt_template = PromptTemplate(input_variables = ["context", "question"], template = qa_template)
    return prompt_template

qa_prompt_template = create_qa_prompt_template()
context = "This video explains the fundamentals of quantum physics."
question = "What are the key principles discussed in the video?"
generated_prompt = qa_prompt_template.format(context = context, question = question)
print(generated_prompt)

# Function to create an LLMChain for question answering
def create_qa_chain(llm, prompt_template, verbose = True):
    return LLMChain(llm = llm, prompt = prompt_template, verbose = verbose)

# Function to retrieve relevant context and generate an answer based on user input
def generate_answer(question, faiss_index, qa_chain, k = 7):
    relevant_context = retrieve(question, faiss_index, k = k)
    answer = qa_chain.predict(context = relevant_context, question = question)
    return answer

# Function to summarizing a video
processed_transcript = ""

def summarize_video(video_url):
    global fetched_transcript, processed_transcript
  
    if video_url:
    		fetched_transcript = get_transcript(video_url)
    		processed_transcript = process(fetched_transcript)
  	else:
  		return "Please provide a valid YouTube URL."

    if processed_transcript:
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
        summary = summary_chain.run({"transcript" : processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

# Function to answer user's questions
def answer_question(video_url, user_question):
    global fetched_transcript, processed_transcript

    if not processed_transcript:
        if video_url:
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        chunks = chunk_transcript(processed_transcript)
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."

# Setting Up Gradio Interface
with gr.Blocks() as interface:
    video_url = gr.Textbox(label = "YouTube Video URL", placeholder = "Enter the YouTube Video URL")
    summary_output = gr.Textbox(label = "Video Summary", lines = 5)
    question_input = gr.Textbox(label = "Ask a Question About the Video", placeholder = "Ask your question")
    answer_output = gr.Textbox(label = "Answer to Your Question", lines = 5)

    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    transcript_status = gr.Textbox(label = "Transcript Status", interactive = False)

    summarize_btn.click(summarize_video, inputs = video_url, outputs = summary_output)
    question_btn.click(answer_question, inputs = [video_url, question_input], outputs = answer_output)

interface.launch(server_name = "0.0.0.0", server_port = 7860)


# # Import necessary libraries for the YouTube bot
# import gradio as gr
# import re  #For extracting video id
# from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
# from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  # For specifying model types
# from ibm_watsonx_ai import APIClient, Credentials  # For API client and credentials management
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
# from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
# from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
# from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  # For retrieving model specifications
# from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings
# from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
# from langchain.chains import LLMChain  # For creating chains of operations with LLMs
# from langchain.prompts import PromptTemplate  # For defining prompt templates
 
# def get_video_id(url):    
#     # Regex pattern to match YouTube video URLs
#     pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
#     match = re.search(pattern, url)
#     return match.group(1) if match else None
 
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import (
#     TranscriptsDisabled,
#     NoTranscriptFound,
#     VideoUnavailable,
#     IpBlocked
# )

# import yt_dlp

# def get_transcript(url):
#     video_id = get_video_id(url)
#     video_url = url

#     ydl_opts = {
#         "skip_download": True,
#         "writesubtitles": True,
#         "writeautomaticsub": True,
#         "subtitleslangs": ["en"],
#         "quiet": True,
#     }

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(video_url, download=False)

#             subtitles = info.get("subtitles") or info.get("automatic_captions")

#             if not subtitles or "en" not in subtitles:
#                 return None

#             # grab first English subtitle entry
#             subtitle_url = subtitles["en"][0]["url"]

#             import requests
#             response = requests.get(subtitle_url)
#             return response.text

#     except Exception as e:
#         print(f"yt-dlp error: {e}")
#         return None
 
 
# def process(transcript):
#     """
#     Works with both:
#     - youtube_transcript_api format (objects/dicts)
#     - yt-dlp subtitle string format
#     """

#     txt = ""

#     if not transcript:
#         return ""

#     # Case 1: yt-dlp returns plain string (SRT/XML)
#     if isinstance(transcript, str):
#         return transcript

#     # Case 2: list of dicts (yt-dlp parsed or manual format)
#     for i in transcript:
#         try:
#             if isinstance(i, dict):
#                 txt += f"Text: {i.get('text', '')} Start: {i.get('start', '')}\n"

#             else:
#                 # youtube_transcript_api format
#                 txt += f"Text: {i.text} Start: {i.start}\n"

#         except Exception:
#             continue

#     return txt
 
# def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
#     # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
 
#     # Split the transcript into chunks
#     chunks = text_splitter.split_text(processed_transcript)
#     return chunks
 
 
# def setup_credentials():
#     # Define the model ID for the WatsonX model being used
#     model_id = "ibm/granite-8b-code-instruct"
   
#     # Set up the credentials by specifying the URL for IBM Watson services
#     credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
   
#     # Create an API client using the credentials
#     client = APIClient(credentials)
   
#     # Define the project ID associated with the WatsonX platform
#     project_id = "skills-network"
   
#     # Return the model ID, credentials, client, and project ID for later use
#     return model_id, credentials, client, project_id
 
# def define_parameters():
#     # Return a dictionary containing the parameters for the WatsonX model
#     return {
#         # Set the decoding method to GREEDY for generating text
#         GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
       
#         # Specify the maximum number of new tokens to generate
#         GenParams.MAX_NEW_TOKENS: 900,
#     }
 
 
# def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
#     # Create and return an instance of the WatsonxLLM with the specified configuration
#     return WatsonxLLM(
#         model_id=model_id,          # Set the model ID for the LLM
#         url=credentials.get("url"),      # Retrieve the service URL from credentials
#         project_id=project_id,            # Set the project ID for accessing resources
#         params=parameters                  # Pass the parameters for model behavior
#     )
 
 
 
# def setup_embedding_model(credentials, project_id):
#     # Create and return an instance of WatsonxEmbeddings with the specified configuration
#     return WatsonxEmbeddings(
#         model_id='ibm/slate-30m-english-rtrvr-v2',  # Set the model ID for the SLATE-30M embedding model
#         url=credentials["url"],                            # Retrieve the service URL from the provided credentials
#         project_id=project_id                               # Set the project ID for accessing resources in the Watson environment
#     )
 
 
 
# def create_faiss_index(chunks, embedding_model):
#     """
#     Create a FAISS index from text chunks using the specified embedding model.
   
#     :param chunks: List of text chunks
#     :param embedding_model: The embedding model to use
#     :return: FAISS index
#     """
#     # Use the FAISS library to create an index from the provided text chunks
#     return FAISS.from_texts(chunks, embedding_model)
 
 
 
# def perform_similarity_search(faiss_index, query, k=3):
#     """
#     Search for specific queries within the embedded transcript using the FAISS index.
   
#     :param faiss_index: The FAISS index containing embedded text chunks
#     :param query: The text input for the similarity search
#     :param k: The number of similar results to return (default is 3)
#     :return: List of similar results
#     """
#     # Perform the similarity search using the FAISS index
#     results = faiss_index.similarity_search(query, k=k)
#     return results
 
 
# def create_summary_prompt():
#     """
#     Create a PromptTemplate for summarizing a YouTube video transcript.
   
#     :return: PromptTemplate object
#     """
#     # Define the template for the summary prompt
#     template = """
#     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
 
#     Instructions:
#     1. Summarize the transcript in a single concise paragraph.
#     2. Ignore any timestamps in your summary.
#     3. Focus on the spoken content (Text) of the video.
 
#     Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
#     Please summarize the following YouTube video transcript:
 
#     {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """
   
#     # Create the PromptTemplate object with the defined template
#     prompt = PromptTemplate(
#         input_variables=["transcript"],
#         template=template
#     )
   
#     return prompt
 
 
# def create_summary_chain(llm, prompt, verbose=True):
#     """
#     Create an LLMChain for generating summaries.
   
#     :param llm: Language model instance
#     :param prompt: PromptTemplate instance
#     :param verbose: Boolean to enable verbose output (default: True)
#     :return: LLMChain instance
#     """
#     return LLMChain(llm=llm, prompt=prompt, verbose=verbose)
 
 
# def retrieve(query, faiss_index, k=7):
#     """
#     Retrieve relevant context from the FAISS index based on the user's query.
 
#     Parameters:
#         query (str): The user's query string.
#         faiss_index (FAISS): The FAISS index containing the embedded documents.
#         k (int, optional): The number of most relevant documents to retrieve (default is 3).
 
#     Returns:
#         list: A list of the k most relevant documents (or document chunks).
#     """
#     relevant_context = faiss_index.similarity_search(query, k=k)
#     return relevant_context
 
# def create_qa_prompt_template():
#     """
#     Create a PromptTemplate for question answering based on video content.
#     Returns:
#         PromptTemplate: A PromptTemplate object configured for Q&A tasks.
#     """
   
#     # Define the template string
#     qa_template = """
#     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
#     1. Precise and free from repetition
#     2. Consistent with the information provided in the video
#     3. Well-organized and easy to understand
#     4. Focused on addressing the user's question directly
#     If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
#     Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>
 
#     <|start_header_id|>user<|end_header_id|>
#     Relevant Video Context: {context}
#     Based on the above context, please answer the following question:
#     {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """
#     # Create the PromptTemplate object
#     prompt_template = PromptTemplate(
#         input_variables=["context", "question"],
#         template=qa_template
#     )
#     return prompt_template
 
 
# def create_qa_chain(llm, prompt_template, verbose=True):
#     """
#     Create an LLMChain for question answering.
 
#     Args:
#         llm: Language model instance
#             The language model to use in the chain (e.g., WatsonxGranite).
#         prompt_template: PromptTemplate
#             The prompt template to use for structuring inputs to the language model.
#         verbose: bool, optional (default=True)
#             Whether to enable verbose output for the chain.
 
#     Returns:
#         LLMChain: An instantiated LLMChain ready for question answering.
#     """
   
#     return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)
 
 
# def generate_answer(question, faiss_index, qa_chain, k=7):
#     """
#     Retrieve relevant context and generate an answer based on user input.
 
#     Args:
#         question: str
#             The user's question.
#         faiss_index: FAISS
#             The FAISS index containing the embedded documents.
#         qa_chain: LLMChain
#             The question-answering chain (LLMChain) to use for generating answers.
#         k: int, optional (default=3)
#             The number of relevant documents to retrieve.
 
#     Returns:
#         str: The generated answer to the user's question.
#     """
 
#     # Retrieve relevant context
#     relevant_context = retrieve(question, faiss_index, k=k)
 
#     # Generate answer using the QA chain
#     answer = qa_chain.predict(context=relevant_context, question=question)
 
#     return answer
 
 
# # Initialize an empty string to store the processed transcript after fetching and preprocessing
# processed_transcript = ""
 
# def summarize_video(video_url):
#     """
#     Title: Summarize Video
 
#     Description:
#     This function generates a summary of the video using the preprocessed transcript.
#     If the transcript hasn't been fetched yet, it fetches it first.
 
#     Args:
#         video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
 
#     Returns:
#         str: The generated summary of the video or a message indicating that no transcript is available.
#     """
#     global fetched_transcript, processed_transcript
   
   
#     if video_url:
#         # Fetch and preprocess transcript
#         fetched_transcript = get_transcript(video_url)
#         if not fetched_transcript:
#             return "Transcript unavailable (YouTube blocked the request or captions not available)."
#         processed_transcript = process(fetched_transcript)
#     else:
#         return "Please provide a valid YouTube URL."
 
#     if processed_transcript:
#         # Step 1: Set up IBM Watson credentials
#         model_id, credentials, client, project_id = setup_credentials()
 
#         # Step 2: Initialize WatsonX LLM for summarization
#         llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
 
#         # Step 3: Create the summary prompt and chain
#         summary_prompt = create_summary_prompt()
#         summary_chain = create_summary_chain(llm, summary_prompt)
 
#         # Step 4: Generate the video summary
#         summary = summary_chain.run({"transcript": processed_transcript})
#         return summary
#     else:
#         return "No transcript available. Please fetch the transcript first."
 
 
# def answer_question(video_url, user_question):
#     """
#     Title: Answer User's Question
 
#     Description:
#     This function retrieves relevant context from the FAISS index based on the user’s query
#     and generates an answer using the preprocessed transcript.
#     If the transcript hasn't been fetched yet, it fetches it first.
 
#     Args:
#         video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
#         user_question (str): The question posed by the user regarding the video.
 
#     Returns:
#         str: The answer to the user's question or a message indicating that the transcript
#              has not been fetched.
#     """
#     global fetched_transcript, processed_transcript
 
#     # Check if the transcript needs to be fetched
#     if not processed_transcript:
#         if video_url:
#             # Fetch and preprocess transcript
#             fetched_transcript = get_transcript(video_url)
#             if not fetched_transcript:
#                 return "Transcript unavailable (YouTube blocked the request or captions not available)."
#             processed_transcript = process(fetched_transcript)
#         else:
#             return "Please provide a valid YouTube URL."
 
#     if processed_transcript and user_question:
#         # Step 1: Chunk the transcript (only for Q&A)
#         chunks = chunk_transcript(processed_transcript)
 
#         # Step 2: Set up IBM Watson credentials
#         model_id, credentials, client, project_id = setup_credentials()
 
#         # Step 3: Initialize WatsonX LLM for Q&A
#         llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
 
#         # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
#         embedding_model = setup_embedding_model(credentials, project_id)
#         faiss_index = create_faiss_index(chunks, embedding_model)
 
#         # Step 5: Set up the Q&A prompt and chain
#         qa_prompt = create_qa_prompt_template()
#         qa_chain = create_qa_chain(llm, qa_prompt)
 
#         # Step 6: Generate the answer using FAISS index
#         answer = generate_answer(user_question, faiss_index, qa_chain)
#         return answer
#     else:
#         return "Please provide a valid question and ensure the transcript has been fetched."
 
 
 
# with gr.Blocks() as interface:
 
#     gr.Markdown(
#         "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
#     )
 
#     # Input field for YouTube URL
#     video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
   
#     # Outputs for summary and answer
#     summary_output = gr.Textbox(label="Video Summary", lines=5)
#     question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
#     answer_output = gr.Textbox(label="Answer to Your Question", lines=5)
 
#     # Buttons for selecting functionalities after fetching transcript
#     summarize_btn = gr.Button("Summarize Video")
#     question_btn = gr.Button("Ask a Question")
 
#     # Display status message for transcript fetch
#     transcript_status = gr.Textbox(label="Transcript Status", interactive=False)
 
#     # Set up button actions
#     summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
#     question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)
 
# # Launch the app with specified server name and port
# interface.launch(server_name="0.0.0.0", server_port=7860)
