import pandas as pd
from utils import clean_text
import json
import os
from collections import Counter
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILES = [
    {"name": "labeled_emotion_data.csv", "column": "text"},
    {"name": "data_en.csv", "column": "lyrics"}
]

TOTAL_VOCAB_TARGET = 30000
MIN_FREQ = 5 



def build_english_vocab():
    master_word_counter = Counter()
    
    print(f"--- Saf İngilizce Sözlük Oluşturuluyor ---")
    print(f"Hedef Kelime Kapasitesi: {TOTAL_VOCAB_TARGET}\n")

    for file_info in FILES:
        file_name = file_info["name"]
        text_column = file_info["column"]
        file_path = os.path.join(DATA_DIR, file_name)
        
        print(f"-> İşleniyor: {file_name}")
        
        try:
            with tqdm(desc=f"{file_name} Parçaları", unit=" chunk") as pbar:
                for chunk in pd.read_csv(file_path, chunksize=10000, usecols=[text_column]):
                    chunk = chunk.dropna(subset=[text_column])
                    
                    for text in chunk[text_column]:
                        cleaned = clean_text(text)
                        words = cleaned.split()
                        master_word_counter.update(words)
                    
                    pbar.update(1)
                    
        except Exception as e:
            print(f"HATA ({file_name}): {e}\n")
            continue

    print(f"\n--- Kelime Sayımı Tamamlandı ---")
    
    filtered = [(w, c) for w, c in master_word_counter.most_common() if c >= MIN_FREQ]
    master_word_set = {w for w, _ in filtered[:TOTAL_VOCAB_TARGET]}
    added_count = len(master_word_set)
            
    print(f"Minimum {MIN_FREQ} kez geçen en popüler {added_count} İngilizce kelime seçildi.")

    vocab = {
        "<PAD>": 0,  
        "<UNK>": 1,  
        "<SOS>": 2,  
        "<EOS>": 3   
    }
    
    idx = 4
    for word in sorted(list(master_word_set)):
        vocab[word] = idx
        idx += 1
            
    print(f"Final Sözlük Boyutu (Özel Tokenler Dahil): {len(vocab)}")
    
    output_path = os.path.join(DATA_DIR, "vocab.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
        
    print(f"BAŞARILI: 'vocab.json' kaydedildi: {output_path}")

if __name__ == "__main__":
    build_english_vocab()