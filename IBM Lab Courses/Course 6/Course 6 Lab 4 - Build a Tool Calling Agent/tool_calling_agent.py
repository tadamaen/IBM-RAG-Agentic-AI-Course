# INSTALLING AND IMPORTING PACKAGES
# %%capture
# %pip install pytube 
# %pip install youtube-transcript-api==1.1.0
# %pip install langchain-community==0.3.16
# %pip install langchain==0.3.23
# %pip install langchain-openai==0.3.14
# %pip install yt-dlp

import re
import json
from pytube import YouTube, Search
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage, ToolMessage
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)

llm = init_chat_model("gpt-4o-mini", model_provider = "openai")

# Defining video ID extraction tool
@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"

print(extract_video_id.name)
print("----------------------------")
print(extract_video_id.description)
print("----------------------------")
print(extract_video_id.func)

# Testing tool execution
extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A")
tools = []
tools.append(extract_video_id)

# Defining transcript fetching tool
@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages = [language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"

fetch_transcript.run("hfIUstzHs9A")
tools.append(fetch_transcript)

# Defining YouTube search tool
@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query (str): The search term to look for on YouTube
        
    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [{"title": yt.title,
                 "video_id": yt.video_id,
                 "url": f"https://youtu.be/{yt.video_id}"} for yt in s.results]
    except Exception as e:
        return f"Error: {str(e)}"

search_out = search_youtube.run("Generative AI")
display(JSON(search_out))
tools.append(search_youtube)

# Defining metadata extraction tool
@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download = False)
        return {'title': info.get('title'),
                'views': info.get('view_count'),
                'duration': info.get('duration'),
                'channel': info.get('uploader'),
                'likes': info.get('like_count'),
                'comments': info.get('comment_count'),
                'chapters': info.get('chapters', [])}

meta_data = get_full_metadata.run("https://www.youtube.com/watch?v=T-D1OfcDW1M")
display(JSON(meta_data))
tools.append(get_full_metadata)

# Defining thumbnail retrieval tool
@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download = False)
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({"url": t['url'], "width": t.get('width'), "height": t.get('height'), "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')})
            return thumbnails
    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]

thumbnails = get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")
display(JSON(thumbnails))
tools.append(get_thumbnails)

# Binding Tools
llm_with_tools = llm.bind_tools(tools)
for tool in tools:
    schema = {"name": tool.name,
              "description": tool.description,
              "parameters": tool.args_schema.schema() if tool.args_schema else {},
              "return": tool.return_type if hasattr(tool, "return_type") else None}
    display(JSON(schema))

# Calling The Tool
query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"
messages = [HumanMessage(content = query)]
response_1 = llm_with_tools.invoke(messages)
messages.append(response_1)

# Creating A Tool Mapping Dictionary
tool_mapping = {"get_thumbnails" : get_thumbnails,
                "extract_video_id": extract_video_id,
                "fetch_transcript": fetch_transcript,
                "search_youtube": search_youtube,
                "get_full_metadata": get_full_metadata}

tool_calls_1 = response_1.tool_calls
display(JSON(tool_calls_1))
tool_name = tool_calls_1[0]['name']
print(tool_name)
tool_call_id = tool_calls_1[0]['id']
print(tool_call_id)
args = tool_calls_1[0]['args']
print(args)

my_tool = tool_mapping[tool_calls_1[0]['name']]
video_id = my_tool.invoke(tool_calls_1[0]['args'])
messages.append(ToolMessage(content = video_id, tool_call_id = tool_calls_1[0]['id']))
response_2 = llm_with_tools.invoke(messages)
messages.append(response_2)
tool_calls_2 = response_2.tool_calls
fetch_transcript_tool_output = tool_mapping[tool_calls_2[0]['name']].invoke(tool_calls_2[0]['args'])
messages.append(ToolMessage(content = fetch_transcript_tool_output, tool_call_id = tool_calls_2[0]['id']))
summary = llm_with_tools.invoke(messages)

# Automating the tool calling process - Extracting tool information from LLM response
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(content = str(result), tool_call_id = tool_call["id"])
    except Exception as e:
        return ToolMessage(content = f"Error: {str(e)}", tool_call_id = tool_call["id"])

# Building The Summarization Chain
summarization_chain = (
    # Start with initial query
    RunnablePassthrough.assign(messages = lambda x: [HumanMessage(content = x["query"])])
    # First LLM call (extract video ID)
    | RunnablePassthrough.assign(ai_response = lambda x: llm_with_tools.invoke(x["messages"]))
    # Process first tool call
    | RunnablePassthrough.assign(tool_messages = lambda x: [execute_tool(tc) for tc in x["ai_response"].tool_calls])
    # Update message history
    | RunnablePassthrough.assign(messages = lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"])
    # Second LLM call (fetch transcript)
    | RunnablePassthrough.assign(ai_response2 = lambda x: llm_with_tools.invoke(x["messages"]))
    # Process second tool call
    | RunnablePassthrough.assign(tool_messages2 = lambda x: [execute_tool(tc) for tc in x["ai_response2"].tool_calls])
    # Final message update
    | RunnablePassthrough.assign(messages = lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"])
    # Generate final summary
    | RunnablePassthrough.assign(summary = lambda x: llm_with_tools.invoke(x["messages"]).content)
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"]))

result = summarization_chain.invoke({"query": "Summarize this YouTube video: https://www.youtube.com/watch?v=1bUy-1hGZpI"})
print("Video Summary:\n", result)

# Creating the initial message setup
initial_setup = RunnablePassthrough.assign(messages = lambda x: [HumanMessage(content=x["query"])])

# Defining the first LLM interaction
first_llm_call = RunnablePassthrough.assign(ai_response = lambda x: llm_with_tools.invoke(x["messages"]))

# Processing the first tool call
first_tool_processing = RunnablePassthrough.assign(tool_messages = lambda x: [execute_tool(tc) for tc in x["ai_response"].tool_calls]).assign(messages = lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"])

# Defining the second LLM interaction
second_llm_call = RunnablePassthrough.assign(ai_response2 = lambda x: llm_with_tools.invoke(x["messages"]))

# Processing the second tool call
second_tool_processing = RunnablePassthrough.assign(tool_messages2 = lambda x: [execute_tool(tc) for tc in x["ai_response2"].tool_calls]).assign(messages = lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"])

# Generating the final summary
final_summary = RunnablePassthrough.assign(summary = lambda x: llm_with_tools.invoke(x["messages"]).content) | RunnableLambda(lambda x: x["summary"])

# Assembling the complete chain
chain = (initial_setup | first_llm_call | first_tool_processing | second_llm_call | second_tool_processing | final_summary)

query = {"query": "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"}
result = summarization_chain.invoke(query)
print("Video Summary:\n", result)

# Testing the Chain with a Different Query
query = {"query": "Get top 3 youtube videos in India and their metadata"}
try:
    result = summarization_chain.invoke(query)
    print("Video Summary:\n", result)
except Exception as e:
    print("Non-critical network error:", e)

# Recursive chain flow
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        content = f"Error: {str(e)}"
    return ToolMessage(content = content, tool_call_id = tool_call["id"])

# Defining the core processing logic
def process_tool_calls(messages):
    """Recursive tool call processor"""
    last_message = messages[-1]
    tool_messages = [execute_tool(tc) for tc in getattr(last_message, 'tool_calls', [])]
    updated_messages = messages + tool_messages
    next_ai_response = llm_with_tools.invoke(updated_messages)
    return updated_messages + [next_ai_response]

# Creating the recursive stopping condition
def should_continue(messages):
    """Check if you need another iteration"""
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))

# Implementing the recursive function
def _recursive_chain(messages):
    """Recursively process tool calls until completion"""
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages
recursive_chain = RunnableLambda(_recursive_chain)

# Building the complete universal chain
universal_chain = (RunnableLambda(lambda x: [HumanMessage(content = x["query"])])
                 | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
                 | recursive_chain)

query_us = {"query": "Show top 3 US trending videos with metadata and thumbnails"}
try:
    response = universal_chain.invoke(query_us)
    print("\nUS Trending Videos:\n", response[-1])
except Exception as e:
    print("Non-critical network error while fetching US trending videos:", e)
