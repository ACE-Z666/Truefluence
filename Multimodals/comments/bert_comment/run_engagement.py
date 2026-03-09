import torch
from torch import nn

# IMPORTANT: model structure must EXACTLY match training
model = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load("models/engagement_mlp.pt"))
model.eval()

def analyze_engagement(followers, likes, comments):
    followers = followers / 100000
    likes = likes / 10000
    comments = comments / 1000

    x = torch.tensor([[followers, likes, comments]], dtype=torch.float32)

    score = model(x).item()

    # Stretch score to make differences visible
    return min(max(score * 1.5, 0), 1)


