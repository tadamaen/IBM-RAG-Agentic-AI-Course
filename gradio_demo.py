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
