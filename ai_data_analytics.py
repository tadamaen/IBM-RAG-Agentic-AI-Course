# INSTALLING AND IMPORTING PACKAGES
# %pip install langchain-openai==0.3.10 | tail -n 1
# %pip install langchain==0.3.21 | tail -n 1
# %pip install openai==1.68.2 | tail -n 1
# %pip install pandas==2.2.3 | tail -n 1
# %pip install numpy==2.2.4 | tail -n 1
# %pip install matplotlib==3.10.1 | tail -n 1
# %pip install seaborn==0.13.2 | tail -n 1
# %pip install scikit-learn==1.6.1 | tail -n 1

# ! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/N0CceRlquaf9q85PK759WQ/regression-dataset.csv
# ! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7J73m6Nsz-vmojwab91gMA/classification-dataset.csv

import numpy as np
import pandas as pd
import matplotlib
import seaborn
import sklearn
import langchain
import openai
import langchain_openai
import glob
import os
from typing import List, Optional

# IGNORE IF YOU ARE NOT RUNNING LOCALLY
os.environ["OPENAI_API_KEY"] = "your OpenAI API key here"

from langchain_core.tools import tool

@tool
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the local directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]

print("Tool Name:", list_csv_files.name)
print("Tool Description:", list_csv_files.description)
print("Tool Arguments:", list_csv_files.args)

DATAFRAME_CACHE = {}
@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.
    
    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.
    
    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    
    return (f"Loaded datasets: {loaded}\n"
            f"Already cached: {cached}")

from typing import List, Optional,Dict,Any

@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]): 
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]: 
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    summaries = []

    for path in dataset_paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
        df = DATAFRAME_CACHE[path]
        summary = {"file_name": path, "column_names": df.columns.tolist(), "data_types": df.dtypes.astype(str).to_dict()}
        summaries.append(summary)
      
    return summaries

@tool
def call_dataframe_method(file_name: str, method: str) -> str:
   """
   Execute a method on a DataFrame and return the result.
   This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe' 
   on a dataset that has already been loaded and cached using 'preload_datasets'.
   Args:
       file_name (str): The path or name of the dataset in the global cache.
       method (str): The name of the method to call on the DataFrame. Only no-argument 
                     methods are supported (e.g., 'head', 'describe', 'info').
   Returns:
       str: The output of the method as a formatted string, or an error message if 
            the dataset is not found or the method is invalid.
   Example:
       call_dataframe_method(file_name="data.csv", method="head")
   """
   if file_name not in DATAFRAME_CACHE:
       try:
           DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
       except FileNotFoundError:
           return f"DataFrame '{file_name}' not found in cache or on disk."
       except Exception as e:
           return f"Error loading '{file_name}': {str(e)}"
   
   df = DATAFRAME_CACHE[file_name]
   func = getattr(df, method, None)
   if not callable(func):
       return f"'{method}' is not a valid method of DataFrame."
   try:
       result = func()
       return str(result)
   except Exception as e:
       return f"Error calling '{method}' on '{file_name}': {str(e)}"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assumes this global cache is shared
DATAFRAME_CACHE = {}

@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns = [target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the regression target.
    Returns:
        Dict[str, float]: A dictionary with R² score and Mean Squared Error.
    """
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns = [target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {"r2_score": r2, "mean_squared_error": mse}

from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

# Prompt, LLM and Tools
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data science assistant. Use the available tools to analyze CSV files. "
     "Your job is to determine whether each dataset is for classification or regression, based on its structure."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")])
llm = init_chat_model("gpt-4o-mini", model_provider = "openai", streaming = False)
tools = [list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset]

# Construct the tool calling agent
agent = create_openai_tools_agent(llm, tools, prompt)
response = agent.invoke({"input": "Can you tell me about the dataset?", "intermediate_steps": []})
action = response[0]

print("🧠 Agent decided to call a tool:")
print("Tool Name:", action.tool)
print("Tool Input:", action.tool_input)
print("Log:\n", action.log.strip())

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True,handle_parsing_errors = True)
agent_executor.agent.stream_runnable = False

print("📊 Ask questions about your dataset (type 'exit' to quit):")

while True:
    user_input=input(" You:")
    if user_input.strip().lower() in ['exit','quit']:
        print("see ya later")
        break
    result = agent_executor.invoke({"input" : user_input})
    print(f"my Agent: {result['output']}")
