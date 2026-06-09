# INSTALLING AND IMPORTING LIBRARIES
# %%capture
# %pip install fastmcp==2.12.2
# %pip install langchain==0.3.27
# %pip install langchain_mcp_adapters==0.1.9
# %pip install langgraph==0.6.7
# %pip install langchain_openai==0.3.33

import socket
import asyncio
from fastmcp import FastMCP, Client


# Setting Up Path Directory and PORT [NOT SO IMPT]
import os
def make_dir():
    if os.path.exists("path"):
        print("✓ Path directory already exists")
    else:
        print("✗ Path directory doesn't exist - creating it...")
        os.makedirs("path")
        print("✓ Path directory created")

PORT = 8000

def test_port(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

f"Port {PORT} is available: {not test_port()}"

def print_stream_info(read, write, _sid, verbose = False):
    """Print information about the read/write streams and session ID."""
    if verbose:
        print("READ (receives FROM server):")
        print(read)
        print()
        
        print("WRITE (sends TO server):")
        print(write)
        print()
        
        print("SESSION ID:")
        print(_sid())

# Setting Up Tools
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

print(multiply.name)
print(multiply.description)
print(multiply.args)

print("What is 2 x 3?")
print("Answer: " + str(multiply.invoke({"a": 2, "b": 3})))

# Creating a Calculator MCPServer (Using FastMCP Object)
from fastmcp import FastMCP
mcp = FastMCP(name = "CalculatorMCPServer",
              instructions = """This server provides data analysis tools.
                                Call get_average() to analyze numerical data.""")
print('mcp object', mcp)

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers together.
       Args: a (int): The first integer, b (int): The second integer
       Returns: int: The sum of `a` and `b`
       Example: add(3, 5) -> 8"""
    return a + b

@mcp.tool
def subtract(a: int, b: int) -> int:
    """Subtract one integer from another.
       Args: a (int): The number to subtract from, b (int): The number to subtract
       Returns: int: The result of `a - b`
       Example: subtract(10, 4) -> 6"""
    return a - b

# Resources
@mcp.resource("file:///endpoint/{name}")
def return_template_document(name: str) -> str:
    """Read a document by name"""
    return f"Document contents of {name}"

make_dir()
%%capture
!wget -P path/ https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/aNE__JjH4DLNEibuNpfDlg/examples.txt
!wget -P path/ https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tfoeGPInNoajVS0DSohdVg/README.txt

@mcp.resource("file://endpoint2/{name}")
def read_document(name: str) -> str:
    """Read a document by name from the path directory"""
    try:
        with open(f"path/{name}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Document '{name}' not found in path directory"
    except Exception as e:
        return f"Error reading document: {str(e)}"

# Prompts
@mcp.prompt(title = "Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

# Creating a Client - In-Memory Transport
from fastmcp import Client
client = Client(mcp)
print(f"client: {client}")

# Using async to call the functions and testing it
async def call_add_tool(a: int, b: int):
    async with client:
        result = await client.call_tool("add", {"a": a, "b": b})
        return result

response = await call_add_tool(4, 5)
print(response.data)
print(response.content[0].text)
print(response.structured_content)

async def call_subtract_tool(a: int, b: int):
    async with client:
        result = await client.call_tool("subtract", {"a": a, "b": b})
        return result

response = await call_subtract_tool(4, 5)
print(response.content[0].text)

# Obtaining The Tools Stored
async with client:
    tools = await client.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

tool_obj = tools[0]
print(tool_obj)

# Input and Output Schemas
input_schema = tool.inputSchema
print(input_schema)
output_schema = tool.outputSchema
print(output_schema)

# Reading Resources
async def call_resource(name):
    async with client:
        result = await client.read_resource(f"file:///endpoint/{name}")
        return result

response = await call_resource("README.txt")
print(response[0].text)

async def call_resource2(name):
    async with client:
        result = await client.read_resource(f"file://endpoint2/{name}")
        return result

response = await call_resource2("README.txt")
response = await call_resource2("random.txt")
resource = response[0]
print(f"uri: {resource.uri}")
print(f"mimeType: {resource.mimeType}")
print(f"meta: {resource.meta}")
print(f"text: {resource.text}")

# Calling Prompts
async def call_prompt(code):
    async with client:
        result = await client.get_prompt("review_code", {"code": code})
        return result
      
response = await call_prompt("CODE TO BE REVIEWED")
message = response.messages[0]
print(f"Prompt Role: {message.role}")
print(f"Prompt Content: {message.content.text}")

# HTTP Transport MCP Servers

# Starting HTTP MCP Server
asyncio.create_task(mcp.run_http_async(port = PORT))
print(f"HTTP MCP Server started in background on port {PORT}")

# HTTP Transport And Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
transport_http = StreamableHttpTransport(url = f"http://127.0.0.1:{PORT}/mcp")
http_client = Client(transport_http)
print('http_client', http_client)

async def test_client_http(client: Client, a: int, b: int) -> int:
    async with client:  
        result = await client.call_tool("add", {"a": a, "b": b})
        return result

response = await test_client_http(http_client, 4, 5)
print(response.content[0].text)

async def get_tool_list(client: Client):
    async with client:
        abstools = await client.list_tools()
        return abstools
        
tool_list = await get_tool_list(http_client)
print(tool_list)

# LangChain Tools with HTTP MCP Servers
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from mcp import ClientSession
llm = "openai:gpt-5-nano"

from mcp.client.streamable_http import streamablehttp_client
async with streamablehttp_client(f"http://127.0.0.1:{PORT}/mcp") as (read, write, _sid):
    print_stream_info(read, write, _sid, verbose = True)

async with streamablehttp_client(f"http://127.0.0.1:{PORT}/mcp") as (read, write, _sid):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        agent = create_react_agent(model = llm, tools = tools)
        agent_response = await agent.ainvoke({"messages": "Use the add tool to add 2 and 1 and let me know if you used a tool."})
print(agent_response['messages'][-1].content)

# STDIO Transport in MCP Servers
server_content = '''from fastmcp import FastMCP

                    mcp = FastMCP(name = "CalculatorMCPServer",
                                  instructions = """This server provides data analysis tools.
                                                    Call add() to add two numbers.""")

                    @mcp.tool
                    def add(a: int, b: int) -> int:
                        """Adds two integer numbers together."""
                        return a + b

                    @mcp.tool
                    def subtract(a:int, b:int) -> int:
                        """Subtracts b from a"""
                        return a - b

                    @mcp.resource("file://documents/{name}")
                    def read_document(name: str) -> str:
                        """Read a document by name"""
                        return "Document contents of {name}"

                    @mcp.prompt(title = "Code Review")
                    def review_code(code: str) -> str:
                        return f"Please review this code: {code}"

                    if __name__ == "__main__":
                        mcp.run()'''

with open('stdio_server.py', 'w') as f:
    f.write(server_content)
print("Created corrected stdio_server.py file")

# Configuring STDIO Transport
transport_stdio = StdioTransport(command = "python",
                                 args = ["stdio_server.py"])

# Creating A STDIO Transport Client
stdio_transport_client = Client(transport_stdio)
print(stdio_transport_client)

# Testing The Client
async def test_client(client: Client, a: int, b: int):
    async with client:
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": a, "b": b})
        return result

response = await test_client(stdio_transport_client, 4, 5)
print(response.content[0].text)

# LangChain Tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command = "python", args = ["stdio_server.py"])

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        agent = create_react_agent(model = llm, tools = tools)
        agent_response = await agent.ainvoke({"messages": "Use the add tool to add 2 and 1 ."})
print(agent_response['messages'][-1].content)

# Multiple MCP Servers (Combination!)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient({"stdio-client": {"command": "python", "args": ["stdio_server.py"], "transport": "stdio"},
                               "http-client": {"url": f"http://127.0.0.1:{PORT}/mcp", "transport": "streamable_http"}})

tools = await client.get_tools()
[tool.name for tool in tools]

# Creating An Agent
llm = "openai:gpt-5-nano"
agent = create_react_agent(model = llm, tools = tools)
agent_response = await agent.ainvoke({"messages": "whats 8 + 7? use tools"})

for i in agent_response['messages']:
    if isinstance(i, HumanMessage):
        message_type = "HUMAN"
    elif isinstance(i, AIMessage):
        message_type = "AI"
    elif isinstance(i, ToolMessage):
        message_type = "TOOL"
    else:
        message_type = "OTHER"

    if i.content == '':
        i.content = "tool call"
    print(f"[{message_type}] {i.content}")
