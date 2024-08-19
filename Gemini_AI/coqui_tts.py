import torch  # Import the PyTorch library for handling computations on GPU or CPU
from TTS.api import TTS  # Import the TTS module for text-to-speech synthesis
import gradio as gr  # Import Gradio, a library for creating web interfaces (not used in this example)
import torch_directml

# Determine the DirectML device (for AMD GPU or compatible devices)
device = torch_directml.device(torch_directml.default_device())

# Load the TTS model with the specified name
tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch')

# Move the TTS model to the specified device (GPU or CPU)
tts.to(device)

# Verify if the model is moved to the GPU
print(f"Model is on device: {device}")

# Define a function to generate audio from text
def generate_audio(text="A journey of a thousand miles begins with a single step."):
    # Generate and save the audio file from the input text
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    # Return the path to the saved audio file
    return "outputs/output.wav"

# Call the function to test it and print the path to the generated audio file
audio_path = generate_audio()
print(audio_path)

# Additionally, print whether the GPU is being used or not
if "privateuseone" in str(device):
    print("The GPU (via DirectML) is being utilized.")
else:
    print("The GPU is not being utilized.")
