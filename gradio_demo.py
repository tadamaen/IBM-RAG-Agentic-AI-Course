# Task 1: Create A Demo Sum Calculator
import gradio as gr
from huggingface_hub import HfFolder

def add_numbers(Num1, Num2):
    return Num1 + Num2

demo = gr.Interface(
    fn = add_numbers, 
    inputs = [gr.Number(), gr.Number()], 
    outputs = gr.Number() 
)

demo.launch(server_name = "127.0.0.1", server_port = 7860)


# Task 2: Combining Two Input Sentences Together

import gradio as gr 
from huggingface_hub import HfFolder

def combine(a, b): 
  return a + " " + b

demo = gr.Interface(
  fn = combine, 
  inputs = [gr.Textbox(label = "Input 1"), gr.Textbox(label = "Input 2") ], 
  outputs = gr.Textbox(label = "Output") 
) 

demo.launch(server_name = "127.0.0.1", server_port = 7860)

# Task 3: Sentence Builder (Using Multiple Inputs With Different Gradio Interface Inputs)

import gradio as gr

def sentence_builder(quantity, tech_worker_type, countries, place, activity_list, morning):
    return f"""The {quantity} {tech_worker_type}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"""

demo = gr.Interface(
    fn = sentence_builder,
    inputs = [gr.Slider(3, 20, value = 4, step = 1, label = "Number Of Workers", info = "Choose between 3 and 20"),
              gr.Dropdown(["Data Scientist", "Software Developer", "Software Engineer"], label = "Tech Worker Type", info = "Will add more tech worker types later!"),
              gr.CheckboxGroup(["Canada", "Japan", "France"], label = "Countries", info = "Where are they from?"),
              gr.Radio(["Office", "Restaurant", "Meeting Room"], label = "Location", info = "Where did they go?"),
              gr.Dropdown(["Partied", "Brainstormed", "Coded", "Fixed Bugs"], value = ["Brainstormed", "Coded"], multiselect = True, label = "Activities", info = "Which activities did they perform?"),
              gr.Checkbox(label = "Morning", info = "Did they carry out the activities in the morning?")],
    outputs = "text",
    examples = [[3, "Software Developer", ["Canada", "Japan"], "Restaurant", ["Coded", "Fixed Bugs"], True],
                [4, "Data Scientist", ["Japan"], "Office", ["Brainstormed", "Partied"], False],
                [10, "Software Engineer", ["Canada", "France"], "Meeting Room", ["Brainstormed"], False],
                [8, "Data Scientist", ["France"], "Restaurant", ["Coded"], True]]
)

demo.launch(server_name="127.0.0.1", server_port= 7860)
