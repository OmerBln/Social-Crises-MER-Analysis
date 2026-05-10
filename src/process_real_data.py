import pandas as pd
import torch
import json
from utils import clean_text
import os
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
CSV_PATH = os.path.join(DATA_DIR, "labeled_emotion_data.csv")
MAX_SEQ_LEN = 128



def process():
    print("Eğitim verisi işleniyor...")
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    PAD_IDX = vocab["<PAD>"]
    UNK_IDX = vocab["<UNK>"]
    SOS_IDX = vocab["<SOS>"]
    EOS_IDX = vocab["<EOS>"]
    
    def text_to_ids(text):
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            return [PAD_IDX] * MAX_SEQ_LEN
            
        words = cleaned_text.split()
        
        token_ids = [vocab.get(w, UNK_IDX) for w in words[:MAX_SEQ_LEN - 2]]
        token_ids = [SOS_IDX] + token_ids + [EOS_IDX]
        
        if len(token_ids) < MAX_SEQ_LEN:
            token_ids += [PAD_IDX] * (MAX_SEQ_LEN - len(token_ids))
            
        return token_ids
    
    df = pd.read_csv(CSV_PATH)
    
    df = df.dropna(subset=['text', 'label'])
    
    print("Kelimeler sözlükle (Word2Vec uyumlu) eşleştiriliyor")
    df['ids'] = df['text'].apply(text_to_ids)
    
    X = torch.tensor(df['ids'].tolist(), dtype=torch.long)
    y = torch.tensor(df['label'].tolist(), dtype=torch.long)
    
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    torch.save((train_x, train_y), os.path.join(DATA_DIR, "real_train.pt"))
    torch.save((val_x, val_y), os.path.join(DATA_DIR, "real_val.pt"))
    
    print(f"\nİşlem Tamam!")
    print(f"Eğitim Seti Boyutu: {len(train_x)}")
    print(f"Validasyon Seti Boyutu: {len(val_x)}")

if __name__ == "__main__":
    process()