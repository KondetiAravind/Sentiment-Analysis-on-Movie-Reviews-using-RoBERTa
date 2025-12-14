import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

MODEL_DIR = "model"
NUM_CLASSES = 5
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model Definition --------
class RobertaClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# -------- Load tokenizer --------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)

# -------- Load model --------
model = RobertaClassifier(NUM_CLASSES)
model.load_state_dict(
    torch.load(f"{MODEL_DIR}/roberta_sentiment.pt", map_location=device)
)
model.to(device)
model.eval()

LABEL_MAP = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

def predict(text: str):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": float(probs[0][pred])
    }

