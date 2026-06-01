# INSTALLING AND IMPORTING PACKAGES
# pip install virtualenv
# virtualenv .venv
# source .venv/bin/activate

# pip install langgraph==0.6.6
# pip install langchain==0.3.27
# pip install langchain-openai==0.3.32
# pip install langchain-ibm==0.3.18
# pip install langchain-mcp-adapters==0.1.9
# touch main.py

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI 
from langchain_ibm import ChatWatsonx

async def main():
    """Main function that sets up and runs an AI agent with access to multiple MCP servers.
       The agent can access Context7 library documentation and Met Museum collections."""
  
    client = MultiServerMCPClient({
            # Context7 server - provides access to library documentation
            "context7": {"url": "https://mcp.context7.com/mcp",     
                         "transport": "streamable_http"},
            # Met Museum server - provides access to museum collection data
            "met-museum": {"command": "npx",                    
                           "args": ["-y", "metmuseum-mcp"],   
                           "transport": "stdio"}})

  	# Initialize the OpenAI language model
    openai_model = ChatOpenAI(model = "gpt-5-nano")

	  # Initialize the Watsonx language model
    watsonx_model = ChatWatsonx(model_id = "ibm/granite-8b-code-instruct", url = "https://us-south.ml.cloud.ibm.com", project_id = "skills-network")
    
    # Retrieve all available tools from the configured MCP servers
    tools = await client.get_tools()

    # Set up conversation memory using InMemorySaver
    checkpointer = InMemorySaver()

	  # Configuration for conversation persistence
    config = {"configurable": {"thread_id": "conversation_id"}}

  	# Create the ReAct agent with all components
    agent = create_react_agent(model = openai_model, tools = tools, checkpointer = checkpointer)

  	# Send initial message to introduce the agent and its capabilities
    response = await agent.ainvoke(
        {"messages": [{"role": "system", "content": "You are a smart, useful agent with tools to access code library documentation and the Met Museum collection."},
                      {"role": "user", "content": "Give a brief introduction of what you do and the tools you can access."}]},
         config = config)
    print(response['messages'][-1].content)

  	# Main interaction loop - allows continuous conversation with the agent
    while True:
        # Display menu options to the user
        choice = input("""Menu:
                          1. Ask the agent a question
                          2. Quit
                          Enter your choice (1 or 2): """)

        if choice == "1":
            print("Your question")
            query = input("> ")
            response = await agent.ainvoke({"messages": query}, config = config)
            print(response['messages'][-1].content)
          
        else:
            print("Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(main())
