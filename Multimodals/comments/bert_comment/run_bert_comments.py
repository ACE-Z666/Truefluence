import torch
import re
import os
from transformers import BertTokenizer, BertModel
from torch import nn

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# -------- Model Definition (same as training) --------
class BERTCommentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False

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

# -------- Load tokenizer & model --------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERTCommentModel()
model_path = os.path.join(os.path.dirname(script_dir), "models", "bert_comment_head.pt")
model.load_state_dict(torch.load(model_path))
model.eval()

# -------- Prediction function --------
def analyze_comments_bert(comments):
    predictions = []
    
    for text in comments:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

    positive = sum(1 for p in predictions if p == 1)
    negative = sum(1 for p in predictions if p == 0)
    total = len(comments)
    
    return {
        "total_comments": total,
        "positive_comments": positive,
        "negative_comments": negative,
        "comment_authenticity_score": round(positive / total, 2) if total > 0 else 0
    }

def get_prediction(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        return torch.argmax(logits, dim=1).item()

# -------- TEST HERE --------
if __name__ == "__main__":
   comments = [
   "Amazing quality for the price",
   "Very happy with the purchase decision",
   "Worst product I have ever bought",
   "Works as expected and more",
   "Stopped working after few days",
   "Completely disappointed",
   "Nice product bro",
   "nice Sadhanam",
   "Fantastic",
   "Fake",
   ]
result = analyze_comments_bert(comments)
print(result)