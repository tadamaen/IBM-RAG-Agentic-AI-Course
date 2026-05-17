import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
	pipe = pipeline("automatic-speech-recognition", model = "openai/whisper-tiny.en", chunk_length_s = 30)
	result = pipe(audio_file, batch_size = 8)["text"]
	return result
  
# Set up Gradio interface
audio_input = gr.Audio(sources = "upload", type = "filepath")  
output_text = gr.Textbox()  
iface = gr.Interface(fn = transcript_audio, inputs = audio_input, outputs = output_text, title = "Audio Transcription App",
		                 description = "Upload the audio file")
iface.launch(server_name = "0.0.0.0", server_port = 5000)
