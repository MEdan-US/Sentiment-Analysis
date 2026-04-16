import os
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from tokenizers import Tokenizer


idx2label = {0: 'negative', 1:'positive'}

def preprocess_text(text):
    # remove URLs https://www.
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    # remove HTML Tags: <>
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    # normalize whitespace
    text = " ".join(text.split())

    # lowercasing
    text = text.lower()
    return text

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, dropout_rate=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 1. Thêm Dropout sau Embedding để tránh phụ thuộc quá mức vào từ vựng cụ thể
        self.embedding_dropout = nn.Dropout(0.2)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k),
                nn.BatchNorm1d(num_filters), # 2. Thêm Batch Norm để ổn định việc học
                nn.ReLU()
            ) for k in kernel_sizes
        ])

        # 3. Dropout mạnh trước khi vào lớp phân loại (Cực kỳ quan trọng)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)

        embedded = self.embedding(x) # (batch_size, seq_len, embed_dim)
        embedded = self.embedding_dropout(embedded)

        # Conv1d yêu cầu đầu vào: (batch_size, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Chạy qua các lớp Conv + Pooling
        conved = [conv(embedded) for conv in self.convs] # List of (batch, num_filters, L_out)

        # Max pooling trên toàn bộ chiều dài câu
        pooled = [F.max_pool1d(c, c.size(-1)).squeeze(-1) for c in conved] # List of (batch, num_filters)

        # Ghép các đặc trưng lại
        cat = self.dropout(torch.cat(pooled, dim=1)) # (batch, len(kernel_sizes) * num_filters)

        return self.fc(cat)

def load_model(model_path, vocab_size=10000, embedding_dim=100, num_classes=2):
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=num_classes
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def inference(sentence_text, tokenizer, model, max_seq_len, device):
    # Preprocess the input sentence using the same function as for training data
    processed_sentence = preprocess_text(sentence_text)

    # Encode the sentence using the trained tokenizer
    encoded = tokenizer.encode(processed_sentence)
    input_ids = encoded.ids

    # Pad or truncate the input_ids to max_seq_len
    if len(input_ids) < max_seq_len:
        padding_needed = max_seq_len - len(input_ids)
        input_ids = input_ids + [tokenizer.token_to_id("<pad>")] * padding_needed
    elif len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    # Convert to tensor, add batch dimension, and move to device
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(input_tensor)

    # Apply softmax to get probabilities
    preds = F.softmax(predictions, dim=1)
    p_max, yhat = torch.max(preds.data, 1)

    # Return predicted probability (as percentage) and class label
    predicted_class_prob = p_max.item()
    predicted_class_label = yhat.item()

    return predicted_class_prob * 100, predicted_class_label
  
model = load_model('text_cnn_model.pt')
tokenizer = Tokenizer.from_file('tokenizer_vi.json')
max_seq_len = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
  st.title('Sentiment Analysis')
  st.title('Model: Text CNN. Dataset: NTC-SCV')
  text_input = st.text_input("Sentence: ", "Đồ ăn ở quán này quá tệ luôn!")
  p, idx = inference(text_input, tokenizer, model, max_seq_len, device)
  label = idx2label[idx]
  st.success(f'Sentiment: {label} with {p:.2f} % probability.') 

if __name__ == '__main__':
     main() 
