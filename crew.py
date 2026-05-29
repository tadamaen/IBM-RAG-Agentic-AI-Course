# TERMINAL COMMANDS
# git clone --no-checkout https://github.com/HaileyTQuach/Smart-Nutritional-App.git NourishBot
# cd NourishBot
# git checkout 1-start\python3.11 -m venv venv
# source venv/bin/activate # activate venv
# pip install -r requirements.txt

import os
import yaml
import base64
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.tools import ExtractIngredientsTool, FilterIngredientsTool, DietaryFilterTool, NutrientAnalysisTool
from ibm_watsonx_ai import Credentials, APIClient
from src.models import RecipeSuggestionOutput, NutrientAnalysisOutput 

credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com", api_key = "<YOUR_API_KEY>")  # Insert Your API Key Here
client = APIClient(credentials)
project_id = "skills-network"
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

# Implementing The Class BaseNourishBotCrew
@CrewBase
class BaseNourishBotCrew:
    agents_config_path = os.path.join(CONFIG_DIR, 'agents.yaml')
    tasks_config_path = os.path.join(CONFIG_DIR, 'tasks.yaml')
    
    def __init__(self, image_data, dietary_restrictions: str = None):
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions

        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

    # Agent 1: Ingredient Detection Agent: Detects ingredients from the uploaded food image.
    @agent
    def ingredient_detection_agent(self) -> Agent:
        return Agent(config = self.agents_config['ingredient_detection_agent'],
                     tools = [ExtractIngredientsTool.extract_ingredient, FilterIngredientsTool.filter_ingredients],
                     allow_delegation = False,
                     max_iter = 5,
                     verbose = True)
      
    # Agent 2: Dietary Filtering Agent: Filters ingredients based on dietary restrictions.
    @agent
    def dietary_filtering_agent(self) -> Agent:
        return Agent(config = self.agents_config['dietary_filtering_agent'],
                     tools = [DietaryFilterTool.filter_based_on_restrictions],
                     allow_delegation = True,
                     max_iter = 6,
                     verbose = True)

    # Agent 3: Nutrient Analysis Agent: Analyzes the nutrients in the detected ingredients.
    @agent
    def nutrient_analysis_agent(self) -> Agent:
        return Agent(config = self.agents_config['nutrient_analysis_agent'],
                     tools = [NutrientAnalysisTool.analyze_image],
                     allow_delegation = False,
                     max_iter = 4,
                     verbose = True)
      
    # Agent 4: Recipe Suggestion Agent: Generates recipe ideas using the filtered ingredients.
    @agent
    def recipe_suggestion_agent(self) -> Agent:
        return Agent(config = self.agents_config['recipe_suggestion_agent'],
                     allow_delegation = False,
                     verbose = True)

    # Task 1: Ingredient Detection Task (From Agent 1 -> Ingredient Detection Agent)
    @task
    def ingredient_detection_task(self) -> Task:
        task_config = self.tasks_config['ingredient_detection_task']
        return Task(description = task_config['description'],
                    agent = self.ingredient_detection_agent(),
                    expected_output = task_config['expected_output'])
      
    # Task 2: Dietary Filtering Task (From Agent 2 -> Dietary Filtering Agent)
    @task
    def dietary_filtering_task(self) -> Task:
        task_config = self.tasks_config['dietary_filtering_task']
        return Task(description = task_config['description'],
                    agent = self.dietary_filtering_agent(),
                    depends_on = ['ingredient_detection_task'],
                    input_data = lambda outputs: {'ingredients': outputs['ingredient_detection_task'],
                                                  'dietary_restrictions': self.dietary_restrictions},
                    expected_output = task_config['expected_output'])

    # Task 3: Nutrient Analysis Task (From Agent 3 -> Nutrient Analysis Agent)
    @task
    def nutrient_analysis_task(self) -> Task:
        task_config = self.tasks_config['nutrient_analysis_task']
        return Task(description = task_config['description'],
                    agent = self.nutrient_analysis_agent(),
                    expected_output = task_config['expected_output'],
                    output_json = NutrientAnalysisOutput)
      
    # Task 4: Recipe Suggestion Task (From Agent 4 -> Recipe Suggestion Agent)
    @task
    def recipe_suggestion_task(self) -> Task:
        task_config = self.tasks_config['recipe_suggestion_task']
        return Task(description = task_config['description'],
                    agent = self.recipe_suggestion_agent(),
                    depends_on = ['dietary_filtering_task'],
                    input_data = lambda outputs: {'filtered_ingredients': outputs['dietary_filtering_task']},
                    expected_output = task_config['expected_output'],
                    output_json = RecipeSuggestionOutput)

# Implementing The Class NourishBotRecipeCrew
@CrewBase
class NourishBotRecipeCrew(BaseNourishBotCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [self.ingredient_detection_task(), self.dietary_filtering_task(), self.recipe_suggestion_task()]
        agents = [self.ingredient_detection_agent(), self.dietary_filtering_agent(), self.recipe_suggestion_agent()]
        return Crew(agents = agents, tasks = tasks, process = Process.sequential, verbose = True)

# Implementing The Class NourishBotAnalysisCrew
@CrewBase
class NourishBotAnalysisCrew(BaseNourishBotCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [self.nutrient_analysis_task()]
        agents = [self.nutrient_analysis_agent()]
        return Crew(agents = agents, tasks = tasks, process = Process.sequential, verbose = True)

# WHEN RUNNING THE APPLICATION
# git reset --hard
# git checkout 5-final
# git pull
# python app.py
