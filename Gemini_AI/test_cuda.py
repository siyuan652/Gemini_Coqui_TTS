import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CUDA is {'available' if device == 'cuda' else 'not available'}. Using {device}.")