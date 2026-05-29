# INSTALLING AND IMPORTING LIBRARIES
# %%capture
# !pip install langchain-openai==0.3.27
# !pip install langgraph==0.6.6
# !pip install pygraphviz==1.14
# !pip install litellm

import litellm
litellm.ssl_verify = False

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field
import operator
from pprint import pprint
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the language model - using gpt-4o-mini for cost-effective experimentation
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# Orchestrator-Worker Pattern (Kitchen Dishes)
# Dish schema for a single dish
class Dish(BaseModel):
    name: str = Field(description = "Name of the dish (for example, Spaghetti Bolognese, Chicken Curry).")
    ingredients: List[str] = Field(description = "List of ingredients needed for this dish, separated by commas.")
    location: str = Field(description = "The cuisine or cultural origin of the dish (for example, Italian, Indian, Mexican).")

# Dishes schema for a list of Dish objects
class Dishes(BaseModel):
    sections: List[Dish] = Field(description = "A list of grocery sections, one for each dish, with ingredients.")

# Construct a prompt template
dish_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that generates a structured grocery list.\n\n"
               "The user wants to prepare the following meals: {meals}\n\n"
               "For each meal, return a section with:\n"
               "- the name of the dish\n"
               "- a comma-separated list of ingredients needed for that dish.\n"
               "- the cuisine or cultural origin of the food")])

# Use LCEL to pipe the prompt to an LLM with a structured output of Dishes
planner_pipe = dish_prompt | llm.with_structured_output(Dishes)
planner_pipe.invoke({"meals": ["banana smoothie", "carrot cake"]})

class State(TypedDict):
    meals: str
    sections: List[Dish]
    completed_menu: Annotated[List[str], operator.add]
    final_meal_guide: str

# Create a dummy state to test our worker nodes
dummy_state: State = {"meals": "Spaghetti Bolognese and Chicken Stir Fry",
                      "sections": [],
                      "completed_menu": [],
                      "final_meal_guide": ""}
report_sections = planner_pipe.invoke({"meals": dummy_state['meals']})

for i, section in enumerate(report_sections.sections):
    print(f"Dish {i+1}\n")
    dummy_state["sections"].append(section)
    print(f"Item Name: {section.name}")
    print(f"Location/Cuisine: {section.location}")
    print(f"Ingredients: {", ".join(section.ingredients)}.")

# Orchestrator Node
def orchestrator(state: State):
    """Orchestrator that generates a structured dish list from the given meals."""
    dish_descriptions = planner_pipe.invoke({"meals": state["meals"]})
    return {"sections": dish_descriptions.sections}

chef_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class chef from {location}.\n\n"
               "Please introduce yourself briefly and present a detailed walkthrough for preparing the dish: {name}.\n"
               "Your response should include:\n"
               "- Start with hello with your  name and culinary background\n"
               "- A clear list of preparation steps\n"
               "- A full explanation of the cooking process\n\n"
               "Use the following ingredients: {ingredients}.")])

chef_pipe = chef_prompt | llm

class WorkerState(TypedDict):
    section: Dish
    completed_menu: Annotated[list, operator.add]

def assign_workers(state: State):
    """Assign a worker to each section in the plan"""
    return [Send("chef_worker", {"section": s}) for s in state["sections"]]

# Defining The Chef Worker
def chef_worker(state: WorkerState):
    """Worker node that generates the cooking instructions for one meal section."""
    meal_plan = chef_pipe.invoke({"name": state["section"].name,
                                  "location": state["section"].location,
                                  "ingredients": state["section"].ingredients})
    return {"completed_menu": [meal_plan.content]}

dummy_dishes: List[Dish] = dummy_state["sections"]
for section in dummy_dishes:
    worker_state: WorkerState = {"section": section,
                                 "recipe": []}
    result = chef_worker(worker_state)
    dummy_state["completed_menu"] += result["completed_menu"]

completed_menu_sections = "\n".join(dummy_state["completed_menu"])
print(completed_menu_sections[:1000])

def synthesizer(state: State):
    """Synthesize full report from sections"""
    completed_sections = state["completed_menu"]
    completed_menu = "\n\n---\n\n".join(completed_sections)
    return {"final_meal_guide": completed_menu}

# Building The Workflow
orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("chef_worker", chef_worker)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["chef_worker"])
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_edge("chef_worker", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)
orchestrator_worker = orchestrator_worker_builder.compile()
display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

# Testing The Workflow
state = orchestrator_worker.invoke({"meals": "Steak and eggs, tacos, and chili"})
pprint(state["final_meal_guide"][:2000])

#-------------------------------------------------------------------------------------------------------
# Reflection Pattern
grades = Literal["ultra-conservative", "conservative", "moderate", "aggressive", "high risk"]

class State(TypedDict):
    investment_plan: str
    investor_profile: str
    target_grade: grades
    feedback: str
    grade: grades
    n: int = 0

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an investment advisor. Given the investor’s profile and their proposed plan,"
               "choose exactly one risk classification from: ultra-conservative, conservative, moderate, aggressive, high risk."
               "Return ONLY the grade."),
    ("user", "Investor profile:\n\n{investor_profile}\n\n")])
grade_pipe = grade_prompt | llm

# Function To Determine Target Grade
def determine_target_grade(state: State):
    """Ask the LLM to pick the best-fitting target_grade."""
    response = grade_pipe.invoke({"investor_profile": state["investor_profile"]})
    return {"target_grade": response.content.lower()}

# Initialize empty state except for the user inputted investor profile
dummy_state: State = {"investment_plan": "",
                      "investor_profile": ("Age: 29\n"
                                           "Salary: $110,000\n"
                                           "Assets: $40,000\n"
                                           "Goal: Achieve financial independence by age 45\n"
                                           "Risk tolerance: High"),
                      "target_grade": "",
                      "feedback": "",
                      "grade": "",
                      "n": 0}

# Get target grade
target_grade = determine_target_grade(dummy_state)
dummy_state.update(target_grade)
pprint(dummy_state)

# Phase 1: Initial Generation
# inital generation, no feedback, only based on profile
cathie_wood_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a bold, innovation-driven investment advisor inspired by Cathie Wood.
                  Your goal is to generate a high-conviction, forward-looking investment plan that embraces disruptive technologies,
                  emerging markets, and long-term growth potential. You are not afraid of short-term volatility as long as the upside is transformational.
                  Create an investment strategy tailored to the investor profile below. Prioritize innovation and high-reward opportunities,
                  such as artificial intelligence, biotechnology, blockchain, or renewable energy.
                  Respond with a concise investment plan in paragraph form."""),
    ("human", "Investor profile:\n\n{investor_profile}")])
cathie_wood_pipe = cathie_wood_prompt | llm

# Evaluator output schema
class Feedback(BaseModel):
    grade: grades = Field(description = "Classify the investment based on risk level, ranging from ultra-conservative to high risk.")
    feedback: str = Field(description = "Provide reasoning for the risk classification assigned to the investment suggestion.")

# Phase 2: Adaptive Generation
ray_dalio_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an investment advisor inspired by Ray Dalio's principles but with adaptive strategy generation.
                  Your goal is to create varied, scenario-aware investment plans that respond dynamically to economic conditions,
                  feedback, and the investor's evolving needs. You adapt your recommendations based on previous evaluations.

                  CORE PRINCIPLES:
                  - Environmental diversification across economic regimes (growth/inflation combinations)
                  - Risk parity weighting by volatility, not just dollar amounts
                  - Inflation-aware asset selection with real return focus
                  - Macroeconomic scenario planning and regime identification
                  
                  ADAPTATION RULES based on feedback:
                  - If deemed "too conservative" → Increase growth equity allocation, add emerging markets, consider alternatives
                  - If deemed "too aggressive" → Add defensive assets, increase bond allocation, focus on dividend stocks
                  - If "lacks inflation protection" → Emphasize TIPS, commodities, REITs, international exposure
                  - If "too complex" → Simplify to core ETF strategy with clear rationale
                  - If "insufficient diversification" → Add geographic, sector, or alternative asset exposure
                  
                  ECONOMIC SCENARIO ADJUSTMENTS:
                  - Rising inflation environment → Emphasize commodities, TIPS, real estate, reduce duration
                  - Stagflation concerns → Focus on energy, materials, international markets, inflation hedges
                  - Deflationary risks → Increase government bonds, high-quality corporate bonds, cash positions
                  - Growth acceleration → Favor technology, consumer discretionary, small-cap growth
                  - Economic uncertainty → Balance with "All Weather" approach using multiple asset classes
                  
                  TARGETING 15% RETURNS through:
                  - Strategic overweighting of growth assets during favorable conditions
                  - Tactical allocation adjustments based on economic regime
                  - Alternative investments (REITs, commodities, international) for diversification
                  - Leverage consideration for qualified investors
                  - Regular rebalancing to capture volatility
                  
                  Respond with a clear, actionable investment plan that reflects current economic conditions 
                  and adapts to the specific feedback provided. Vary your approach significantly based on 
                  the grade and feedback received.
                  """),
    ("human","""Investor profile: {investor_profile}
                Previous strategy grade: {grade}
                Evaluator feedback: {feedback}
                Based on this feedback, create a NEW investment strategy that addresses the concerns raised 
                while targeting 15% returns. Make significant adjustments from any previous approach.""")])

ray_dalio_pipe = ray_dalio_prompt | llm

# Build The Generator Node
def investment_plan_generator(state: State) -> dict:
    """Prompts an LLM to generate or improve an investment plan based on the current state."""
    if state.get("feedback"):
        response = ray_dalio_pipe.invoke({"investor_profile": state["investor_profile"],
                                          "grade": state["grade"],
                                          "feedback": state["feedback"]})
    else:
        response = cathie_wood_pipe.invoke({"investor_profile": state["investor_profile"]})
    return {"investment_plan": response.content}

# Get the investment plan
initial_investment_plan = investment_plan_generator(dummy_state)
dummy_state.update(initial_investment_plan)
pprint(dummy_state)

# Evaluator Node

# Warren Buffet style evaluation prompt
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an investment risk evaluator inspired by Warren Buffett's value investing philosophy.
                  Your task is to assess whether a proposed investment strategy aligns with conservative, value-driven principles 
                  that emphasize capital preservation, long-term stability, and sound business fundamentals. You should be 
                  skeptical of speculative investments, high-volatility assets, and short-term market trends.

                  RISK CLASSIFICATION LEVELS:
                  - ultra-conservative: Extremely safe, minimal risk of loss
                  - conservative: Low risk, prioritizes capital preservation  
                  - moderate: Balanced approach with acceptable risk-reward ratio
                  - aggressive: Higher risk for potentially greater returns
                  - high risk: Speculative investments with significant loss potential
                
                  EVALUATION CRITERIA:
                  - Business clarity: Is the investment easily understandable with transparent cash flows?
                  - Margin of safety: Does the investment price provide protection against downside risk?
                  - Capital preservation: Will this strategy protect wealth over the long term?
                  - Investor alignment: Does this match a conservative investor's risk tolerance and goals?
                  - Quality fundamentals: Are the underlying assets financially sound with competitive advantages?

                  Return your assessment in the following format:
                  {{"grade": "<investment risk level>",
                    "feedback": "<concise explanation of the assigned risk level and key reasoning>"}}"""),
    ("human", "Evaluate this investment plan:\n\n{investment_plan}\n\nFor this investor profile:\n\n{investor_profile}\n\nAnd provide feedback that matches this target risk level: {target_grade}")])

buffett_evaluator_pipe = evaluator_prompt | llm.with_structured_output(Feedback)

# Building The Generator Node
def evaluate_plan(state: State):
    """LLM evaluates the investment plan"""
    current_count = state.get('n', 0) + 1
    evaluation_result = buffett_evaluator_pipe.invoke({"investment_plan": state["investment_plan"],
                                                       "investor_profile": state["investor_profile"],
                                                       "target_grade": state["target_grade"]})
    return {"grade": evaluation_result.grade, "feedback": evaluation_result.feedback, "n": current_count}

evaluated_feedback = evaluate_plan(dummy_state)
dummy_state.update(evaluated_feedback)
print(f"Grade: {dummy_state["grade"]}")
print(f"Feedback: {dummy_state["feedback"]}")

# Building The Routing Node
def route_investment(state: State, iteration_limit: int = 5):
    """Route investment based on risk grade evaluation"""
    current_grade = state.get("grade", "MISSING")
    target_grade = state.get("target_grade", "MISSING")
    match = current_grade == target_grade

    print(f"=== ROUTING  ===")
    print(f"Current grade: '{current_grade}'")
    print(f"Target risk profile: '{target_grade}'")
    print(f"Match: {match}")
    print(f"Number of trials: {state['n']}")

    if match:
        print("→ Routing to: Accepted")
        return "Accepted"
    elif state['n'] > iteration_limit:
        print("→ Too many iterations, stopping")
        return "Accepted"
    else:
        print("→ Routing to: Rejected + Feedback")
        return "Rejected + Feedback"

# Building The Workflow
optimizer_builder = StateGraph(State)

optimizer_builder.add_node("determine_target_grade", determine_target_grade)
optimizer_builder.add_node("investment_plan_generator", investment_plan_generator)
optimizer_builder.add_node("evaluate_plan", evaluate_plan)

optimizer_builder.add_edge(START, "determine_target_grade")
optimizer_builder.add_edge("determine_target_grade", "investment_plan_generator")
optimizer_builder.add_edge("investment_plan_generator", "evaluate_plan")

optimizer_builder.add_conditional_edges("evaluate_plan", lambda state: route_investment(state), {"Accepted": END, "Rejected + Feedback": "investment_plan_generator"})

optimizer_workflow = optimizer_builder.compile()
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# TestingThe Workflow
state = optimizer_workflow.invoke({
    "investor_profile": ("Age: 29\n"
                         "Salary: $110,000\n"
                         "Assets: $40,000\n"
                         "Goal: Achieve financial independence by age 45\n"
                         "Risk tolerance: High")})

def pretty_print_final_state(state: dict):
    print("🎯 Final Investment Plan Summary\n" + "="*40)
    print(f"\n📌 Investor Profile:\n{state['investor_profile']}")
    
    print("\n📈 Target Risk Grade:", state['target_grade'])
    print("📊 Final Assigned Grade:", state['grade'])
    print("🔁 Iterations Taken:", state['n'])

    print("\n📝 Evaluator Feedback:\n" + "-"*30)
    print(state['feedback'])

    print("\n📃 Final Investment Plan:\n" + "-"*30)
    print(state['investment_plan'])

pretty_print_final_state(state)
