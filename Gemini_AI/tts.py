import os
import torch
from TTS.api import TTS
import torch_directml

def generate_audio(text, output_path="outputs/output.wav"):
    try:
        # Determine the DirectML device (for AMD GPU or compatible devices)
        device = torch_directml.device(torch_directml.default_device())
        
        # Load the TTS model with the specified name
        tts = TTS(model_name='tts_models/en/ljspeech/fast_pitch')
        
        # Move the TTS model to the specified device (GPU or CPU)
        tts.to(device)
        
        # Verify if the model is moved to the GPU
        print(f"Model is on device: {device}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save the audio file from the input text
        tts.tts_to_file(text=text, file_path=output_path)
        
        # Return the path to the saved audio file
        return output_path
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
