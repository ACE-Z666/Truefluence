import torch #pytorch deep learning framework
import pandas as pd #handle csv data
from torch import nn #neural network layer like linear, relu, dropout
from torch.utils.data import Dataset, DataLoader #dataset -custom dataset classes for comments //dataloader -load data in batches during training 
from transformers import BertTokenizer, BertModel #BERT tokenizer and model 
import os

# used Hugging Face Transformers to leverage pretrained BERT for NLP tasks.

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Load data
csv_path = os.path.join(script_dir, "comments.csv")
df = pd.read_csv(csv_path, header=None, names=["comment", "label"], engine='python')

class CommentDataset(Dataset): #stores comments ,labels,tokerizers
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): #length ~
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )
        return { #returs a dictionary containing input ids, attention mask, and label for the comment at index idx
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

class BERTCommentModel(nn.Module): #bert + custom classifier head (defines the neural network model)
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

            #  Freeze BERT
        for p in self.bert.parameters():
            p.requires_grad = False
            # We freeze BERT to reduce training time and avoid overfitting. Only the classifier is trained.
                         
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# Setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = CommentDataset(
    df["comment"].tolist(),
    df["label"].tolist(),
    tokenizer
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = BERTCommentModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)


# Training
for epoch in range(5):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
model_path = os.path.join(os.path.dirname(script_dir), "models", "bert_comment_head.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print("✅ BERT comment head trained")