# %pip install -q langgraph==0.2.57 langchain-ibm==0.3.10
from langgraph.graph import StateGraph
from typing import TypedDict, Optional

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
