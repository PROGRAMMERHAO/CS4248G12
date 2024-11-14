import torch

model_state = torch.load('../models/hybert_model/model.pt', map_location='cpu')
print("Model loaded successfully.")
