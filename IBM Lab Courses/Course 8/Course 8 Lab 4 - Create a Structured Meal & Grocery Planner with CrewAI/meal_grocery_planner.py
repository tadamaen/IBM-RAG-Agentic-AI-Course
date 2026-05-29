# INSTALLING AND IMPORTING PACKAGES
# %pip install langchain==0.3.20 | tail -n 1
# %pip install crewai==0.141.0 | tail -n 1
# %pip install langchain-community==0.3.19 | tail -n 1
# %pip install langchain-openai==0.3.25 | tail -n 1
# %pip install duckduckgo-search==7.5.2 | tail -n 1
# %pip install crewai-tools==0.51.1 | tail -n 1
# %pip install databricks-sdk==0.46.0 | tail -n 1
# !pip install litellm
# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/3xGOgzMOv5jhRsA3A8N9fQ/leftover.py"

import litellm
import sys
import os
from enum import Enum
from leftover import LeftoversCrew
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from IPython.display import display, JSON, Markdown
from datetime import datetime
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM
from leftover import LeftoversCrew
sys.path.append(".")
files = os.listdir('.')
litellm.ssl_verify = False

# Class 1: GroceryItem (Individual grocery item with details)
class GroceryItem(BaseModel):
    """Individual grocery item"""
    name: str = Field(description = "Name of the grocery item")
    quantity: str = Field(description = "Quantity needed (for example, '2 lbs', '1 gallon')")
    estimated_price: str = Field(description = "Estimated price (for example, '$3-5')")
    category: str = Field(description = "Store section (for example, 'Produce', 'Dairy')")

sample_item = GroceryItem(name = "Chicken Breast",
                          quantity = "2 lbs",
                          estimated_price = "$8-12",
                          category = "Meat")

print("🛒 Sample Grocery Item Structure:")
display(JSON(sample_item.model_dump()))

# Class 2: MealPlan (Recipe information with researched ingredients)
class MealPlan(BaseModel):
    """Simple meal plan"""
    meal_name: str = Field(description = "Name of the meal")
    difficulty_level: str = Field(description = "'Easy', 'Medium', 'Hard'")
    servings: int = Field(description = "Number of people it serves")
    researched_ingredients: List[str] = Field(description = "Ingredients found through research")

sample_meal = MealPlan(meal_name = "Chicken Stir Fry",
                       difficulty_level = "Easy",
                       servings = 4,
                       researched_ingredients = ["chicken breast", "broccoli", "bell peppers", "garlic", "soy sauce", "rice"])

print("\n🍽️ Sample Meal Plan Structure:")
display(JSON(sample_meal.model_dump()))

# Class 3: ShoppingCategory (Store section with organized items)
class ShoppingCategory(BaseModel):
    """Store section with items"""
    section_name: str = Field(description = "Store section (for example, 'Produce', 'Dairy')")
    items: List[GroceryItem] = Field(description = "Items in this section")
    estimated_total: str = Field(description = "Estimated cost for this section")

sample_section = ShoppingCategory(section_name = "Produce",
                                  items = [GroceryItem(name = "Bell Peppers", quantity = "3 pieces", estimated_price = "$3-4", category = "Produce"),
                                           GroceryItem(name = "Onions", quantity = "2 lbs", estimated_price = "$2-3", category = "Produce")],
                                  estimated_total = "$5-7")

print("\n🏪 Sample Shopping Section:")
display(JSON(sample_section.model_dump()))

# Class 4: GroceryShoppingPlan (Complete shopping strategy with budget analysis)
class GroceryShoppingPlan(BaseModel):
    """Complete simplified shopping plan"""
    total_budget: str = Field(description = "Total planned budget")
    meal_plans: List[MealPlan] = Field(description = "Planned meals")
    shopping_sections: List[ShoppingCategory] = Field(description = "Organized by store sections")
    shopping_tips: List[str] = Field(description = "Money-saving and efficiency tips")

# This serves as an example: DO NOT RUN THIS CODE
# weekly_shopping_plan = GroceryShoppingPlan(total_budget = "$40-50",
#                                            meal_plans = [breakfast_meal, lunch_meal, dinner_meal],
#                                            shopping_sections = [dairy_section, meat_section, pantry_section, produce_section],
#                                            shopping_tips = [...])

# Setting Up LLM and Essential Tools
os.environ["WATSONX_API_BASE"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WX_PROJECT_ID"] = "skills-network"
llm = LLM(model = "watsonx/ibm/granite-4-h-small")
os.environ['SERPER_API_KEY'] = 'YOUR_SERPER_API_KEY_HERE'            # Fill in your API key from SerperDev here

# Creating Meal and Grocery Planning Workflow with CrewAI
# Agent 1: Meal Planning Agent (responsible for researching recipes and creating detailed meal plans)
meal_planner = Agent(role = "Meal Planner & Recipe Researcher",
                     goal = "Search for optimal recipes and create detailed meal plans",
                     backstory = "A skilled meal planner who researches the best recipes online, considering dietary needs, cooking skill levels, and budget constraints.",
                     tools = [SerperDevTool()],
                     llm = llm,
                     verbose = False)

meal_planning_task = Task(description = ("Search for the best '{meal_name}' recipe for {servings} people within a {budget} budget. "
                                         "Consider dietary restrictions: {dietary_restrictions} and cooking skill level: {cooking_skill}. "
                                         "Find recipes that match the skill level and provide complete ingredient lists with quantities."),
                          expected_output = "A detailed meal plan with researched ingredients, quantities, and cooking instructions appropriate for the skill level.",
                          agent = meal_planner,
                          output_pydantic = MealPlan,
                          output_file = "meals.json")

meal_planner_crew = Crew(agents = [meal_planner],
                         tasks = [meal_planning_task],
                         process = Process.sequential,
                         verbose = True)

meal_planner_result = meal_planner_crew.kickoff(inputs = {"meal_name": "Chicken Stir Fry",
                                                          "servings": 4,
                                                          "budget": "$25",                           
                                                          "dietary_restrictions": ["no nuts"],       
                                                          "cooking_skill": "beginner"})

print("✅ Single meal planning completed!")
print("📋 Single Meal Results:")
print(meal_planner_result)

# Agent 2: Shopping Organization Agent (responsible for transforming meal plans into organized shopping lists)
shopping_organizer = Agent(role = "Shopping Organizer", 
                           goal = "Organize grocery lists by store sections efficiently",
                           backstory = "An experienced shopper who knows how to organize lists for quick store trips and considers dietary restrictions.",
                           tools = [],
                           llm = llm,
                           verbose = False)

shopping_task = Task(description = ("Organize the ingredients from the '{meal_name}' meal plan into a grocery shopping list. "
                                    "Group items by store sections and estimate quantities for {servings} people. "
                                    "Consider dietary restrictions: {dietary_restrictions} and cooking skill: {cooking_skill}. "
                                    "Stay within budget: {budget}."),
                     expected_output = "An organized shopping list grouped by store sections with quantities and prices.",
                     agent = shopping_organizer,
                     context = [meal_planning_task],
                     output_pydantic = GroceryShoppingPlan,
                     output_file = "shopping_list.json")

two_agent_grocery_crew = Crew(agents = [meal_planner, shopping_organizer],
                              tasks = [meal_planning_task, shopping_task],
                              process = Process.sequential,
                              verbose = True)

shopping_result = two_agent_grocery_crew.kickoff(inputs = {"meal_name": "Chicken Stir Fry",
                                                           "servings": 4,
                                                           "budget": "$25",                           
                                                           "dietary_restrictions": ["no nuts"],      
                                                           "cooking_skill": "beginner"})

print("✅ Complete meal planning + shopping completed!")
print("📋 Shopping Results:")
print(shopping_result)

# Agent 3: Budget Advisor Agent (responsible for watching the budget and providing money-saving advice)
budget_advisor = Agent(role = "Budget Advisor",
                       goal = "Provide cost estimates and money-saving tips",
                       backstory = "A budget-conscious shopper who helps families save money on groceries while respecting dietary needs.",
                       tools = [SerperDevTool()],
                       llm = llm,
                       verbose = False)

budget_task = Task(description = ("Analyze the shopping plan for '{meal_name}' serving {servings} people. "
                                  "Ensure total cost stays within {budget}. Consider dietary restrictions: {dietary_restrictions}. "
                                  "Provide practical money-saving tips and alternative ingredients if needed to meet budget."),
                   expected_output = "A complete shopping guide with detailed prices, budget analysis, and money-saving tips.",
                   agent = budget_advisor,
                   context = [meal_planning_task, shopping_task],
                   output_file = "shopping_guide.md")


# Agent 4: Food Leftover Agent
# PLEASE REFER TO THE LEFTOVER.PY FILE FOR THE CODE 
leftovers_cb = LeftoversCrew(llm = llm)
yaml_leftover_manager = leftovers_cb.leftover_manager()
yaml_leftover_task = leftovers_cb.leftover_task()

# Agent 5: Summary And Task Agent (responsible for gathering all the content and create a detailed summary)
summary_agent = Agent(role = "Report Compiler",
                      goal = "Compile comprehensive meal planning reports from all team outputs",
                      backstory = "A skilled coordinator who organizes information from multiple specialists into comprehensive, easy-to-follow reports.",
                      tools = [],
                      llm = llm,
                      verbose = False)

summary_task = Task(description = ("Compile a comprehensive meal planning report that includes:\n"
                                   "1. The complete recipe and cooking instructions from the meal planner\n"
                                   "2. The organized shopping list with prices from the shopping organizer\n"
                                   "3. The budget analysis and money-saving tips from the budget advisor\n"
                                   "4. The leftover management suggestions from the waste reduction specialist\n"
                                   "Format this as a complete, user-friendly meal planning guide."),
                    expected_output = "A comprehensive meal planning guide that combines all team outputs into one cohesive report.",
                    agent = summary_agent,
                    context = [meal_planning_task, shopping_task, budget_task, yaml_leftover_task])

# Assembling The Complete Grocery Planning Team
complete_grocery_crew = Crew(agents = [meal_planner, shopping_organizer, budget_advisor, yaml_leftover_manager, summary_agent],
                             tasks = [meal_planning_task, shopping_task, budget_task, yaml_leftover_task, summary_task],
                             process = Process.sequential,
                             verbose = True)

# Run the complete crew
complete_result = complete_grocery_crew.kickoff(inputs = {"meal_name": "Chicken Stir Fry",
                                                          "servings": 4,
                                                          "budget": "$25",
                                                          "dietary_restrictions": ["no nuts", "low sodium"],
                                                          "cooking_skill": "beginner"})

print("✅ Complete meal planning with summary completed!")
print("📋 Complete Results:")
print(complete_result)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# EXERCISES: 
# Exercise 1 - Create a Specialized Dietary Agent (specializes in dietary analysis and nutritional recommendations)
nutrition_analyst = Agent(role = "Nutrition Analyst & Health Advisor",
                          goal = "Analyze meal nutritional content and provide healthy recommendations",
                          backstory = "A certified nutritionist who evaluates meals for nutritional balance, calorie content, and health optimization while respecting dietary restrictions.",
                          tools = [SerperDevTool()],
                          llm = llm,
                          verbose = False)

nutrition_task = Task(description = ("Analyze the nutritional content of the '{meal_name}' meal plan for {servings} people. "
                                     "Calculate approximate calories, protein, carbs, and fats. Consider dietary restrictions: {dietary_restrictions}. "
                                     "Provide healthy alternatives if the meal could be more nutritious while staying within {budget}."),
                      expected_output = "Nutritional analysis with calorie estimates, macronutrient breakdown, and healthy improvement suggestions.",
                      agent = nutrition_analyst,
                      context = [meal_planning_task, shopping_task, budget_task],
                      output_file = "nutrition_analysis.md")

health_focused_crew = Crew(agents = [meal_planner, shopping_organizer, budget_advisor, nutrition_analyst, yaml_leftover_manager, summary_agent],
                           tasks = [meal_planning_task, shopping_task, budget_task, nutrition_task, yaml_leftover_task, summary_task],
                           process = Process.sequential,
                           verbose = True)

result = health_focused_crew.kickoff(inputs = {"meal_name": "Quinoa Buddha Bowl",
                                               "servings": 2,
                                               "budget": "$20",
                                               "dietary_restrictions": ["vegetarian", "high protein"],
                                               "cooking_skill": "intermediate"})

# Exercise 2 - Extend Pydantic Models for Weekly Planning (handle weekly meal planning instead of single meals)
class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch" 
    DINNER = "dinner"
    SNACK = "snack"

class DailyMeals(BaseModel):
    """Meals for one day"""
    date: str = Field(description = "Date in YYYY-MM-DD format")
    breakfast: Optional[MealPlan] = Field(default = None, description = "Breakfast meal plan")
    lunch: Optional[MealPlan] = Field(default = None, description = "Lunch meal plan") 
    dinner: Optional[MealPlan] = Field(default = None, description = "Dinner meal plan")
    snacks: Optional[List[MealPlan]] = Field(default = None, description = "Snack meal plans")
    
class WeeklyMealPlan(BaseModel):
    """Complete weekly meal planning"""
    week_start_date: str = Field(description = "Start date of the week")
    daily_meals: List[DailyMeals] = Field(description = "Meals for each day")
    weekly_themes: List[str] = Field(description = "Cooking themes for the week")
    prep_suggestions: List[str] = Field(description = "Meal prep recommendations")

class WeeklyGroceryPlan(BaseModel):
    """Weekly grocery shopping strategy"""
    weekly_budget: str = Field(description = "Total weekly budget")
    meal_plans: List[DailyMeals] = Field(description = "All weekly meals")
    shopping_sections: List[ShoppingCategory] = Field(description = "Organized by store sections")
    bulk_items: List[GroceryItem] = Field(description = "Items to buy in bulk")
    shopping_tips: List[str] = Field(description = "Weekly shopping efficiency tips")
    budget_breakdown: Dict[str, str] = Field(description = "Daily budget allocation")

# Test the models
sample_weekly_plan = WeeklyMealPlan(week_start_date = "2024-01-15",
                                    daily_meals = [DailyMeals(date = "2024-01-15",
                                                              breakfast = MealPlan(meal_name = "Oatmeal", difficulty_level = "Easy", servings = 2, researched_ingredients = ["oats", "milk", "berries"]),
                                                              lunch = MealPlan(meal_name = "Salad", difficulty_level = "Easy", servings = 2, researched_ingredients = ["lettuce", "tomatoes", "dressing"]),
                                                              dinner = MealPlan(meal_name = "Pasta", difficulty_level = "Medium", servings = 2, researched_ingredients = ["pasta", "sauce", "cheese"]))],
                                    weekly_themes = ["Italian Monday", "Taco Tuesday"],
                                    prep_suggestions = ["Wash vegetables on Sunday", "Cook grains in bulk"])

display(JSON(sample_weekly_plan.model_dump()))
