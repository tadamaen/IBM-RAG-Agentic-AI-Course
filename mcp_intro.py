%%capture
%pip install fastmcp

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport, StdioTransport

# STDIN and STDOUT
my_code = "hello"
print(my_code)

import sys
sys.stdout.write("Hello")

name = input("What is your name?")
sys.exit("Notebook exited after input.")
print(name)

# STDERR
print("This is standard output (stdout)")
print("This is an error message (stderr)", file = sys.stderr)

# Example: catching a real error
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error occurred: {e}", file = sys.stderr)

# STDIO Transport
stdio_transport = StdioTransport(command = "npx", args = ["-y", "@upstash/context7-mcp"])
print(stdio_transport)

# STDIO Client
stdio_client = Client(stdio_transport)

async with stdio_client as client:
    tools = await client.list_tools()

print("Done")
len(tools)
print(tools[0].name)
print(tools[0].description)
print(tools[0].inputSchema)

# call_tool function
async with stdio_client as client:
    response = await client.call_tool("resolve-library-id", {"libraryName": "fastmcp",
                                                             "query": "I want to create a new MCP server using the fastmcp Python framework"})
print(response.content[0].text)

async with stdio_client as client:
    docs = await client.call_tool("query-docs", {"libraryId": "/llmstxt/gofastmcp_llms-full_txt",
                                                 "query": "I want to fetch the code snippets and the documentation",
                                                 "tokens": 5000})
print(docs.content[0].text[:1000])

# Exercise 1: How do you use the resolve-library-id tool to find the library ID for scikit-learn?
async with stdio_client as client:
   response = await client.call_tool("resolve-library-id", {"libraryName": "scikit-learn",
                                                            "query": "I want to use scikit-learn package"})
print(response.content[0].text[:1500])

# Exercise 2: How do you get the actual documentation once you have the library ID?
async with stdio_client as client:
    docs = await client.call_tool("query-docs", {"libraryId": "/scikit-learn/scikit-learn",
                                                 "query": "I want to fetch the documentation of the package.",
                                                 "tokens": 5000})
print(docs.content[0].text[:1000])

# HTTP Preface
import requests
url = 'https://www.ibm.com/'
r = requests.get(url)
print(r.status_code)
print(r.request.headers)
print("request body:", r.request.body)

# HTTP Transport
http_transport = StreamableHttpTransport(url = "https://mcp.context7.com/mcp")
http_client = Client(http_transport)

async with http_client as client:
    tools = await client.list_tools()
    response = await client.call_tool("resolve-library-id", {"libraryName": "fastmcp",
                                                             "query": "I want to create a new MCP server using the fastmcp Python framework"})
    docs = await client.call_tool("query-docs", {"libraryId": "/llmstxt/gofastmcp_llms-full_txt",
                                                 "query": "I want to fetch the code snippets and the documentation",
                                                 "tokens": 5000})

for tool in tools:
    print(f"""{tool.name}: {tool.description}, {tool.inputSchema}""")
    print(response.content[0].text[:1000])
    print(docs.content[0].text[:500]) 
