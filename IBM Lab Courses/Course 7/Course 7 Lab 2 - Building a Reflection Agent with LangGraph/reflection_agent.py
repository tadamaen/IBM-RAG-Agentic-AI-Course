# %pip install -q langgraph==0.3.31 
# %pip install -q langchain-ibm==0.3.10
# %pip install -q langchain==0.3.23
# %pip install -q langchain_community==0.3.21 
# %pip install -q pygraphviz==1.14

from langchain_ibm import ChatWatsonx
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph, StateGraph
from typing import List, Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Building an Optimized LinkedIn Post Generator with a Reflection Agent

# Instantiating the Language Model
llm = ChatWatsonx(model_id = "ibm/granite-4-h-small",
                  url = "https://us-south.ml.cloud.ibm.com",
                  project_id = "skills-network")

# Generation Prompt for Posts
generation_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
                " Generate the best LinkedIn post possible for the user's request."
                " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed."),
      MessagesPlaceholder(variable_name = "messages")])

# Creating the Chain for LinkedIn Post Generation
generate_chain = generation_prompt | llm

# Reflection Prompt for LinkedIn Post Critique
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. 
        Follow these guidelines:
        1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.
        
        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """),
    MessagesPlaceholder(variable_name = "messages")])

# Creating the Reflect Chain
reflect_chain = reflection_prompt | llm

# Defining the Agent State for Reflection Agent
from langgraph.graph import MessageGraph
from typing import List, Annotated, TypedDict
from langchain.schema import HumanMessage, AIMessage, SystemMessage
graph = MessageGraph()

# Defining The Generation And Reflection Node
def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    generated_post = generate_chain.invoke({"messages": state})
    return [AIMessage(content = generated_post.content)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content = res.content)]

# Adding Nodes And Edges to the Graph
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")

# Adding A Router Node For Decision Making
def should_continue(state: List[BaseMessage]):
    print(state)
    print(len(state))
    if len(state) > 6:
        return END
    return "reflect"
  
# Adding Conditional Edges
graph.add_conditional_edges("generate", should_continue)

# Compiling And Running Workflow
workflow = graph.compile()
inputs = HumanMessage(content = """Write a linkedin post on getting a software developer job at IBM under 160 characters""")
response = workflow.invoke(inputs)

# Plotting The Graph
from IPython.display import Image, display
display(Image(workflow.get_graph().draw_png()))
