# import os
# import torch
# import torch.nn as nn
# from transformers import DebertaV2Model

# class EmotionClassifier(nn.Module):
#     def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=5):
#         super().__init__()

#         # Load model only from local cache if available
#         self.deberta = DebertaV2Model.from_pretrained(
#             model_name,
#             local_files_only=True  # prevents downloading from HF hub
#         )

#         hidden_size = self.deberta.config.hidden_size
#         self.dropout = nn.Dropout(0.3)
#         self.norm = nn.LayerNorm(hidden_size)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.deberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         cls_hidden_state = outputs.last_hidden_state[:, 0, :]
#         x = self.norm(cls_hidden_state)
#         x = self.dropout(x)
#         logits = self.classifier(x)
#         return logits



import torch
import torch.nn as nn
from transformers import DebertaV2Model

class EmotionClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=5):
        super().__init__()

        # Load pretrained DeBERTa model directly from HF Hub
        # Streamlit will auto-cache this after first download
        self.deberta = DebertaV2Model.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        hidden_size = self.deberta.config.hidden_size

        # Extra layers
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token (first token representation)
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        x = self.norm(cls_hidden)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
