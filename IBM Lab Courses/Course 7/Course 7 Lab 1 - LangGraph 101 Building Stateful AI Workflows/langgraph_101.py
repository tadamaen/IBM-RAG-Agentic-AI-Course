# %pip install -q langgraph==0.2.57 langchain-ibm==0.3.10
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
from langchain_ibm import ChatWatsonx

class AuthState(TypedDict):
    username: Optional[str] 
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]

# Example Objects and Their States
# Object 1: Successful Login
auth_state_1: AuthState = {"username": "alice123",
                           "password": "123",
                           "is_authenticated": True,
                           "output": "Login successful."}
print(f"auth_state_1: {auth_state_1}")
# Object 2: Unsuccessful Login
auth_state_2: AuthState = {"username":"",
                           "password": "wrongpassword",
                           "is_authenticated": False,
                           "output": "Authentication failed. Please try again."}
print(f"auth_state_2: {auth_state_2}")

# Defining The Input Node
def input_node(state):
    print(state)
    if state.get('username', "") == "":
        username = input("What is your username?")
    password = input("Enter your password: ")
    if state.get('username', "") == "":
        return {"username": username, "password": password}
    else:
        return {"password": password}

input_node(auth_state_1)
input_node(auth_state_2)

# Defining The Validate Credentials Node
def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")
    print("Username :", username, "Password :", password)
    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False
    return {"is_authenticated": is_authenticated}

# Defining The Success Node
def success_node(state):
    return {"output": "Authentication successful! Welcome."}

# Defining The Failure Node
def failure_node(state):
    return {"output": "Not Successfull, please try again!"}

# Defining The Router Node
def router(state):
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"

# Creating The Graph
from langgraph.graph import StateGraph
from langgraph.graph import END

workflow = StateGraph(AuthState)

# Adding Nodes To The Graph
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)

# Adding Edges To The Graph
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_edge("Success", END)
workflow.add_edge("Failure", "InputNode")

# Adding ConditionalEdges To The Graph
workflow.add_conditional_edges("ValidateCredential", router, {"success_node": "Success", "failure_node": "Failure"})

# Setting The Entry Point
workflow.set_entry_point("InputNode")

# Compiling The Workflow
app = workflow.compile()

# Running The Application - password is secure_password
inputs = {"username": "test_user"}
result = app.invoke(inputs)
result['output']
print(result)

# Building a QA Workflow Specific to the Guided Project

# Define the structure of the QA state
class QAState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]

# Create an example object
qa_state_example = QAState(question = "What is the purpose of this guided project?",
                           context = "This project focuses on building a chatbot using Python.",
                           answer = None)

for key, value in qa_state_example.items():
    print(f"{key}: {value}")

# Defining The Input Validation Node
def input_validation_node(state):
    question = state.get("question", "").strip()
    if not question:
        return {"valid": False, "error": "Question cannot be empty."}
    return {"valid": True}
input_validation_node(qa_state_example)

# Defining The Context Provider
def context_provider_node(state):
    question = state.get("question", "").lower()
    if "langgraph" in question or "guided project" in question:
        context = ("This guided project is about using LangGraph, a Python library to design state-based workflows. "
                   "LangGraph simplifies building complex applications by connecting modular nodes with conditional edges.")
        return {"context": context}
    return {"context": None}

# Integrating LLM for QA Workflow
llm = ChatWatsonx(model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
                  url = "https://us-south.ml.cloud.ibm.com",
                  project_id = "skills-network")

def llm_qa_node(state):
    question = state.get("question", "")
    context = state.get("context", None)
    if not context:
        return {"answer": "I don't have enough context to answer your question."}
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer the question based on the provided context."
    try:
        response = llm.invoke(prompt)
        return {"answer": response.content.strip()}
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}

# Creating The QA Workflow Graph
qa_workflow = StateGraph(QAState)
qa_workflow.add_node("InputNode", input_validation_node)
qa_workflow.add_node("ContextNode", context_provider_node)
qa_workflow.add_node("QANode", llm_qa_node)
qa_workflow.set_entry_point("InputNode")
qa_workflow.add_edge("ContextNode", "QANode")
qa_workflow.add_edge("QANode", END)
qa_app = qa_workflow.compile()

# Asking An Irrlevant Question
qa_app.invoke({"question": "What is the weather today?"})

# Asking A Relevant Question
qa_app.invoke({"question": "What is LangGraph?"})
qa_app.invoke({"question": "What is the best guided project?"})

# Create A Simple Counter Using LangGraph

# STEP 1: DEFINE THE STATE TYPE
import random
import string
from typing import TypedDict
from langgraph.graph import StateGraph, END

class ChainState(TypedDict):
    n: int
    letter: str

# STEP 2: CREATE ADD NODE FUNCTION
def add(state: ChainState) -> ChainState:
    random_letter = random.choice(string.ascii_lowercase)
    return {**state, "n": state["n"] + 1, "letter": random_letter}

# STEP 3: CREATE PRINTOUT NODE FUNCTION
def print_out(state: ChainState) -> ChainState:
    print("Current n:", state["n"], "Letter:", state["letter"])
    return state

# STEP 4: STOP CONDITION
def stop_condition(state: ChainState) -> bool:
    return state["n"] >= 1000

# STEP 5: GRAPH CONSTRUCTION
workflow = StateGraph(ChainState)
workflow.add_node("add", add)
workflow.add_node("print", print_out)
workflow.add_edge("add", "print")
workflow.add_conditional_edges("print", stop_condition, {True: END, False: "add"})
workflow.set_entry_point("add")

# STEP 6: COMPILE AND RUN
app = workflow.compile()
result = app.invoke({"n": 1, "letter": ""})
