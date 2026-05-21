import torch
import pandas as pd
from torch import nn

df = pd.read_csv("bert_comment/engagement.csv")

X = df[["followers", "likes", "comments"]].values

# NORMALIZATION (same as inference)
X[:, 0] = X[:, 0] / 100000   # followers
X[:, 1] = X[:, 1] / 10000    # likes
X[:, 2] = X[:, 2] / 1000     # comments

y = df["label"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

print("✅ Engagement MLP trained")
torch.save(model.state_dict(), "models/engagement_mlp.pt")