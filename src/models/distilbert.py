"""DistilBERT with a 5-way classification head, fully fine-tuned."""
import torch.nn as nn


class DistilBertRater(nn.Module):
    def __init__(self, name="distilbert-base-uncased", num_classes=5, dropout=0.1):
        super().__init__()
        from transformers import AutoModel  # imported lazily
        self.bert = AutoModel.from_pretrained(name)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.fc(self.drop(out[:, 0]))  # [CLS] token
