# IBM-RAG-Agentic-AI-Course
Contains all the Python code for the labs in the IBM RAG &amp; Agentic AI Course, grouped by course number


## Course 2 Lab 4: LinkedIn Icebreaker Bot

An AI-powered tool that generates personalized icebreakers and conversation starters based on LinkedIn profiles. This project uses IBM watsonx.ai and LlamaIndex to create a tool that helps make introductions more personal and engaging.

### Project Overview

Imagine you're heading to a big networking event, surrounded by potential employers and industry leaders. You want to make a great first impression, but you're struggling to come up with more than the usual, "What do you do?"

This AI icebreaker bot does the research for you. You input a name, and within seconds, the bot searches LinkedIn, generating personalized icebreakers based on someone's career highlights, interests, and even fun facts.

### Features

- Extract LinkedIn profile data using ProxyCurl API or mock data
- Process and index the data using LlamaIndex and IBM watsonx embeddings
- Generate interesting facts about a person's career or education
- Answer specific questions about the LinkedIn profile
- Interact with the bot through a command-line interface or a Gradio web UI

### Project Structure

```
icebreaker_bot/
├── requirements.txt           # Dependencies
├── config.py                  # Configuration settings
├── modules/
│   ├── __init__.py
│   ├── data_extraction.py     # LinkedIn profile data extraction
│   ├── data_processing.py     # Data splitting and indexing
│   ├── llm_interface.py       # LLM setup and interaction
│   └── query_engine.py        # Query processing and response generation
├── app.py                     # Gradio interface
└── main.py                    # Main script to run without Gradio
```

### Getting Started

#### Prerequisites

- Python 3.11+
- A ProxyCurl API key (optional - you can use mock data)

#### Installation

1. Clone the repository:
```bash
git clone https://github.com/HaileyTQuach/icebreaker.git
cd icebreaker
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your ProxyCurl API key to `config.py` (optional)

#### Usage

##### Command Line Interface

Run the bot using the command line:

```bash
python main.py --mock  # Use mock data
# OR
python main.py --url "https://www.linkedin.com/in/username/" --api-key "your-api-key"
```

##### Web Interface

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to the URL shown in the terminal (typically http://127.0.0.1:7860).

### Development Tasks

This is a starter template with placeholder functions. Your task is to implement the following components:

1. In `config.py`:
   - Define the prompt templates for facts generation and question answering

2. In `modules/data_extraction.py`:
   - Implement the `extract_linkedin_profile` function

3. In `modules/data_processing.py`:
   - Implement the `split_profile_data` function
   - Implement the `create_vector_database` function
   - Implement the `verify_embeddings` function

4. In `modules/llm_interface.py`:
   - Implement the `create_watsonx_embedding` function
   - Implement the `create_watsonx_llm` function
   - Implement the `change_llm_model` function

5. In `modules/query_engine.py`:
   - Implement the `generate_initial_facts` function
   - Implement the `answer_user_query` function

6. Update `modules/__init__.py` to import your implemented functions

7. In `main.py`:
   - Implement the `process_linkedin` function
   - Implement the `chatbot_interface` function

8. In `app.py`:
   - Implement the `process_profile` function
   - Implement the `chat_with_profile` function
## Acknowledgments

- IBM watsonx.ai for providing the LLM and embedding models
- LlamaIndex for the data indexing and retrieval framework
- ProxyCurl for LinkedIn profile data extraction

## Course 8 Lab 6: Building your own AI Nutrition Coach using a Multi-Agent System and Multimodal AI

---
title: AI_NutriCoach
app_file: app.py
sdk: gradio
sdk_version: 5.12.0
---
# AI NutriCoach (aka AI Dietary Crew)

AI NutriCoach is an AI-powered nutrition assistant that leverages advanced vision models and natural language processing to detect ingredients from food images, filter ingredients based on dietary restrictions, estimate calories, provide detailed nutrient analysis, and generate recipe suggestions. This project demonstrates the use of CrewAI, WatsonX, and other AI tools to deliver insightful and personalized nutritional feedback.

## Features

- **Ingredient Detection**  
  Detects ingredients from user-uploaded images using a vision AI model.

- **Dietary Filtering**  
  Filters detected ingredients based on user-defined dietary restrictions (e.g., vegan, gluten-free).

- **Calorie Estimation**  
  Estimates total calories from the detected ingredients.

- **Nutrient Analysis**  
  Provides a detailed breakdown of key nutrients such as protein, carbohydrates, fats, vitamins, and minerals.

- **Health Evaluation**  
  Summarizes the overall healthiness of the meal and provides a health evaluation.

- **Recipe Suggestion**  
  Generates recipe ideas based on the filtered ingredients and dietary restrictions.

## How It Works

The project is built using the CrewAI framework, which organizes agents and tasks into workflows for two primary use cases:

1. **Recipe Workflow**  
   Detects ingredients, filters them based on dietary restrictions, and suggests recipes.

2. **Analysis Workflow**  
   Directly estimates calories, performs nutrient analysis, and provides a health evaluation summary from a food image.

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (optional but recommended)
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HaileyTQuach/Smart-Nutritional-App.git
   cd Smart-Nutritional-App
   ```
2. **Create and activate a virtual environment**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. **Install the required dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Create a .env file in the root directory with the following keys**:
   ```bash
    WATSONX_API_KEY=your_watsonx_api_key
    WATSONX_URL=your_watsonx_url
    WATSONX_PROJECT_ID=your_watsonx_project_id
   ```
## Usage
### Run the Application

You can run the application using the following commands:

1. For recipe suggestions

```bash
python main.py <image_path> <dietary_restrictions> recipe
```

Example:

```bash
python main.py food.jpg vegan recipe
```

2. For food analysis

```bash
python main.py <image_path> analysis
```

Example:

```bash
python main.py food.jpg analysis
```

3. For training (future functionality - TODO)

```bash
python main.py train <n_iterations> <output_filename> <image_path> <dietary_restrictions> <workflow_type>
```

## File Structure

```
Smart-Nutritional-App-Crew/
│
├── config/
│   ├── agents.yaml               # Configuration for agents
│   └── tasks.yaml                # Configuration for tasks
│
├── src/
│   ├── crew.py                   # Crew definitions (agents, tasks, workflows)
│   ├── tools.py                  # Tool definitions for ingredient detection, filtering, etc.
│   └── main.py                   # Main script for running the application
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or support, please contact [Hailey Thao Quach](mailto:hailey@haileyq.com).
