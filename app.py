"""Gradio web interface for the Icebreaker Bot."""

import os
import sys
import logging
import uuid
import gradio as gr
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, vector_database
from modules.llm_interface import change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query
import config

# Set up logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers = [logging.StreamHandler(stream = sys.stdout)])
logger = logging.getLogger(__name__)

active_indices = {}

# Function 1: process_profile(linkedin_url, api_key, use_mock, selected_model) -> processes LinkedIn profiles
def process_profile(linkedin_url, api_key, use_mock, selected_model):
    try:
        if selected_model != config.LLM_MODEL_ID:
            change_llm_model(selected_model)

        if use_mock and not linkedin_url:
            linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"
            
        profile_data = extract_linkedin_profile(linkedin_url, api_key if not use_mock else None, mock = use_mock)
        if not profile_data:
            return "Failed to retrieve profile data. Please check the URL or API key.", None

        nodes = split_profile_data(profile_data)
        if not nodes:
            return "Failed to process profile data into nodes.", None
        
        index = create_vector_database(nodes)
        if not index:
            return "Failed to create vector database.", None

        if not verify_embeddings(index):
            logger.warning("Some embeddings may be missing or invalid")
          
        facts = generate_initial_facts(index)
        session_id = str(uuid.uuid4())
        active_indices[session_id] = index
        return f"Profile processed successfully!\n\nHere are 3 interesting facts about this person:\n\n{facts}", session_id
    
    except Exception as e:
        logger.error(f"Error in process_profile: {e}")
        return f"Error: {str(e)}", None

# Function 2: chat_with_profile(session_id, user_query, chat_history) -> handles chatting with the profile
def chat_with_profile(session_id, user_query, chat_history):
    if not session_id:
        return chat_history + [[user_query, "No profile loaded. Please process a LinkedIn profile first."]]
  
    if session_id not in active_indices:
        return chat_history + [[user_query, "Session expired. Please process the LinkedIn profile again."]]
    
    if not user_query.strip():
        return chat_history
    
    try:
        index = active_indices[session_id]
        response = answer_user_query(index, user_query)
        return chat_history + [[user_query, response.response]]
    
    except Exception as e:
        logger.error(f"Error in chat_with_profile: {e}")
        return chat_history + [[user_query, f"Error: {str(e)}"]]

# Creating The Gradio Interface
def create_gradio_interface():
    available_models = ["ibm/granite-4-h-small", "meta-llama/llama-3-2-11b-vision-instruct"]
    
    with gr.Blocks(title = "LinkedIn Icebreaker Bot") as demo:
         gr.Markdown("# LinkedIn Icebreaker Bot")
         gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")
        
         with gr.Tab("Process LinkedIn Profile"):
             with gr.Row():
                 with gr.Column():
                     linkedin_url = gr.Textbox(label = "LinkedIn Profile URL", placeholder = "https://www.linkedin.com/in/username/")
                     api_key = gr.Textbox(label = "ProxyCurl API Key (Leave empty to use mock data)",
                                          placeholder = "Your ProxyCurl API Key",
                                          type = "password",
                                          value = config.PROXYCURL_API_KEY)
                     use_mock = gr.Checkbox(label = "Use Mock Data", value = True)
                     model_dropdown = gr.Dropdown(choices = available_models,
                                                  label = "Select LLM Model",
                                                  value = config.LLM_MODEL_ID)
                     process_btn = gr.Button("Process Profile")
                
                 with gr.Column():
                     result_text = gr.Textbox(label = "Initial Facts", lines = 10)
                     session_id = gr.Textbox(label = "Session ID", visible = False)
            
             process_btn.click(fn = process_profile,
                               inputs = [linkedin_url, api_key, use_mock, model_dropdown],
                               outputs = [result_text, session_id])
        
         with gr.Tab("Chat"):
             gr.Markdown("Chat with the processed LinkedIn profile")
             chatbot = gr.Chatbot(height = 500)
             chat_input = gr.Textbox(label = "Ask a question about the profile",
                                     placeholder = "What is this person's current job title?")
             chat_btn = gr.Button("Send")
            
             chat_btn.click(fn = chat_with_profile,
                            inputs = [session_id, chat_input, chatbot],
                            outputs = [chatbot])
            
             chat_input.submit(fn = chat_with_profile,
                               inputs = [session_id, chat_input, chatbot],
                               outputs = [chatbot])
    
    return demo

# Launch The Interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name = "127.0.0.1",  server_port = 5000, share = False)
