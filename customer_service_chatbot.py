from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai_tools import PDFSearchTool, SerperDevTool
import litellm
litellm.ssl_verify = False

# Download the FAQ document for the tool to use
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf

# Configuring the LLM
llm = LLM(model = "watsonx/ibm/granite-4-h-small",
          base_url = "https://us-south.ml.cloud.ibm.com",
          project_id = "skills-network")

import os
os.environ['SERPER_API_KEY'] = 'YOUR_API_KEY_HERE'       # Insert your Serper API Key using SepperDev
web_search_tool = SerperDevTool()

import warnings
warnings.filterwarnings('ignore')

# Creating The PDF Search Tool
pdf_search_tool = PDFSearchTool(pdf = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf",
                                config = dict(embedder = dict(provider = "huggingface",
                                                              config = dict(model = "sentence-transformers/all-MiniLM-L6-v2"))))

# Approach 1: The Standard Method (Agent-Centric Tools)
agent_centric_agent = Agent(role = "The Daily Dish Inquiry Specialist",
                            goal = """Accurately answer customer questions about The Daily Dish restaurant. 
                                      You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
                            backstory = """You are an AI assistant for 'The Daily Dish'.
                                           You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
                                           Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
                            tools = [pdf_search_tool, web_search_tool],
                            verbose = True,
                            allow_delegation = False,
                            llm = llm)

agent_centric_task = Task(description = "Answer the following customer query: '{customer_query}'. "
                                        "Analyze the question and use the tools at your disposal (PDF search or web search) to find the most relevant information. "
                                        "Synthesize the findings into a clear and friendly response.",
                          expected_output = "A comprehensive and well-formatted answer to the customer's query.",
                          agent = agent_centric_agent)

agent_centric_crew = Crew(agents = [agent_centric_agent],
                          tasks = [agent_centric_task],
                          process = Process.sequential,
                          verbos e= False)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True:
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    if not user_input:
        print("Please type a question.")
        continue
    try:
        result_agent_centric = await agent_centric_crew.kickoff_async(inputs = {'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_agent_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")

# Approach 2: A More Focused Method (Task-Centric Tools)
task_centric_agent = Agent(role = "Customer Service Specialist",
                           goal = "Provide exceptional customer service by following a multi-step process to answer customer questions accurately.",
                           backstory = """You are an AI assistant for 'The Daily Dish'.
                                          You are an expert at following instructions. You will be given a sequence of tasks to complete.
                                          For each task, you will be provided with the specific tool needed to accomplish it.
                                          Your job is to execute each task diligently and pass the results to the next step.""",
                           tools = [],
                           verbose = True,
                           allow_delegation = False,
                           llm = llm)

faq_search_task = Task(description = "Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
                       expected_output = "A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
                       tools = [pdf_search_tool],
                       agent = task_centric_agent)

response_drafting_task = Task(description = "Using the information gathered from the FAQ search, draft a friendly and comprehensive response to the customer's query: '{customer_query}'.",
                              expected_output = "The final, customer-facing response.",
                              agent = task_centric_agent,
                              context = [faq_search_task])

task_centric_crew = Crew(agents = [task_centric_agent],
                         tasks = [faq_search_task, response_drafting_task],
                         process = Process.sequential,
                         verbose = True)

print("\nWelcome to The Daily Dish Chatbot!")
print("What would you like to know? (Type 'exit' to quit)")

while True: 
    user_input = input("\nYour question: ").lower()
    if user_input == 'exit':
        print("Thank you for chatting. Have a great day!")
        break
    if not user_input:
        print("Please type a question.")
        continue
    try:
        result_task_centric = task_centric_crew.kickoff(inputs = {'customer_query': user_input})
        print("\n--- The Daily Dish Assistant ---")
        print(result_task_centric)
        print("--------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")
