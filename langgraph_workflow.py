# INSTALLING AND IMPORTING PACKAGES
# %%capture
# %pip install langchain-openai==0.3.27
# %pip install langgraph==0.6.6
# %pip install pygraphviz==1.14

from langgraph.graph import StateGraph, END, START
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import httpx

# Printing Workflow Information
def print_workflow_info(workflow, app = None):
    """Prints comprehensive information about a LangGraph workflow."""
    print("WORKFLOW INFORMATION")
    print("====================")
    print(f"Nodes: {workflow.nodes}")
    print(f"Edges: {workflow.edges}")
    try:
        finish_points = workflow.finish_points
        print(f"Finish points: {finish_points}")
    except:
        try:
            print(f"Finish point: {workflow._finish_point}")
        except:
            print("Finish points attribute not directly accessible")
    
    if app:
        print("\nWorkflow Visualization:")
        from IPython.display import display
        display(app.get_graph().draw_png())

  client = httpx.Client(verify=False)

llm = ChatOpenAI(model = "gpt-4o-mini", http_client = client)

# Prompt Chaining - Defining The Chain State
class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

# Resume Summsry Agent
def generate_resume_summary(state: ChainState) -> ChainState:
    prompt = f"""You're a resume assistant. 
                 Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.
                 Job Description: {state['job_description']}"""
    response = llm.invoke(prompt)
    return {**state, "resume_summary": response.content}

# Cover Letter Agent
def generate_cover_letter(state: ChainState) -> ChainState:
    prompt = f"""You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.
                 Resume Summary: {state['resume_summary']}
                 Job Description: {state['job_description']}"""
    response = llm.invoke(prompt)
    return {**state, "cover_letter": response.content}
