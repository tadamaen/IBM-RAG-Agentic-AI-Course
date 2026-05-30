# !pip install autogen==0.7 openai==1.64.0 python-dotenv==1.1.0 | tail -n 1

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings('ignore', category = UserWarning)

from autogen import ConversableAgent, GroupChat, GroupChatManager
from openai import OpenAI
import logging
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

client = OpenAI()
code_execution_config = {"use_docker": False}

# Creating Conversable Agents
llm_config = {"config_list": [{"model": "gpt-4", "api_key": None}]}            # Replace with real API key

# Agent 1: Patient Agent (responsible for describing symptoms and requesting medical assistance)
patient_agent = ConversableAgent(name = "patient", 
                                 system_message = "You describe symptoms and ask for medical help.", 
                                 llm_config = llm_config)

# Agent 2: Diagnosis Agent (analyzes the symptoms provided by the patient and generates a concise diagnosis in a single response)
diagnosis_agent = ConversableAgent(name = "diagnosis", 
                                   system_message = "You analyze symptoms and provide a possible diagnosis. Summarize key points in one response.", 
                                   llm_config = llm_config)

# Agent 3: Pharmacy Agent (follows up on the diagnosis by recommending medications)
pharmacy_agent = ConversableAgent(name = "pharmacy", 
                                  system_message = "You recommend medications based on diagnosis. Only respond once.", 
                                  llm_config = llm_config)

# Agent 4: Consultation Agent (provides a final summary of the consultation along with clear next steps)
consultation_agent = ConversableAgent(name = "consultation", 
                                      system_message="""You determine if a doctor's visit is required. Provide a final summary with clear next steps. 
                                                        IMPORTANT: End your response with 'CONSULTATION_COMPLETE' to signal the end of the conversation.""", 
                                      llm_config = llm_config)

# Create GroupChat for Structured Interaction
groupchat = GroupChat(agents = [diagnosis_agent, pharmacy_agent, consultation_agent],
                      messages = [], 
                      max_round = 5,
                      speaker_selection_method = "round_robin")

manager = GroupChatManager(name = "manager", groupchat = groupchat)

# Get Patient Input and Start Consultation
print("\n🤖 Welcome to the AI Healthcare Consultation System!")
symptoms = input("🩺 Please describe your symptoms: ")

print("\n🩺 Diagnosing symptoms...")
response = patient_agent.initiate_chat(manager, message = f"I am feeling {symptoms}. Can you help?")

# EXERCISE: Create a Mental Health Chatbot Using the AutoGen Library
from autogen import ConversableAgent, GroupChat, GroupChatManager
llm_config = {"config_list": [{"model": "gpt-4o", "api_key": None}]}           # Provide OpenAI API key if required

patient_agent = ConversableAgent(name = "patient",
                                 system_message = "You describe your emotions and mental health concerns.",
                                 llm_config = llm_config)

emotion_analysis_agent = ConversableAgent(name = "emotion_analysis",
                                          system_message = "You analyze the user's emotions based on their input."
                                                           "Do not provide treatment or self-care advice."
                                                           "Instead, just summarize the dominant emotions they may be experiencing.",
                                          llm_config = llm_config)

therapy_recommendation_agent = ConversableAgent(name = "therapy_recommendation",
                                                system_message = "You suggest relaxation techniques and self-care methods"
                                                                 "only based on the analysis from the Emotion Analysis Agent."
                                                                 "Do not analyze emotions — just give recommendations based on the prior response.",
                                                llm_config = llm_config)

# Create GroupChat for AI Agents 
groupchat = GroupChat(agents = [emotion_analysis_agent, therapy_recommendation_agent],
                      messages = [], 
                      max_round = 3, 
                      speaker_selection_method = "round_robin")

manager = GroupChatManager(name = "manager", groupchat = groupchat)

# Function to start the chatbot interaction 
def start_mental_health_chat():
    """Runs a chatbot for mental health support with distinct agent roles.""" 
    print("\nWelcome to the AI Mental Health Chatbot!") 
    user_feelings = input("How are you feeling today?")
    print("\nAnalyzing emotions...")
    response = patient_agent.initiate_chat(manager, message = f"I have been feeling {user_feelings}. Can you help?")

    if not response:
        response = therapy_recommendation_agent.initiate_chat(manager, message = "Based on the user's emotions, please provide therapy recommendations.")

# Run the chatbot 
start_mental_health_chat()
