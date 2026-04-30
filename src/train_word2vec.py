import pandas as pd
import re
import os
from gensim.models import Word2Vec
import multiprocessing

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_LYRICS = os.path.join(DATA_DIR, "data_en.csv") 
INPUT_EMOTIONS = os.path.join(DATA_DIR, "labeled_emotion_data.csv")
OUTPUT_MODEL = os.path.join(DATA_DIR, "lyrics_word2vec.model")
OUTPUT_VECTORS = os.path.join(DATA_DIR, "lyrics_vectors.txt")

MAX_CHUNKS = 50 

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class CombinedTextIterator:
    def __init__(self, lyrics_path, emotions_path):
        self.lyrics_path = lyrics_path
        self.emotions_path = emotions_path

    def __iter__(self):
        if os.path.exists(self.emotions_path):
            df_emo = pd.read_csv(self.emotions_path)
            for text in df_emo['text']:
                cleaned_text = clean_text(text)
                if cleaned_text:
                    yield cleaned_text.split()
        
        chunk_count = 0
        if os.path.exists(self.lyrics_path):
            for chunk in pd.read_csv(self.lyrics_path, chunksize=10000, usecols=['lyrics']):
                if chunk_count >= MAX_CHUNKS:
                    break 
                
                for text in chunk['lyrics']:
                    cleaned_text = clean_text(text)
                    if cleaned_text: 
                        yield cleaned_text.split()
            
                chunk_count += 1

def train_fast_embeddings():
    sentences = CombinedTextIterator(INPUT_LYRICS, INPUT_EMOTIONS)
    cores = multiprocessing.cpu_count()
    
    w2v_model = Word2Vec(
        sentences=sentences, 
        vector_size=256,   
        window=5, 
        min_count=5,       
        workers=cores,
        epochs=5           
    )
    
    w2v_model.save(OUTPUT_MODEL)
    w2v_model.wv.save_word2vec_format(OUTPUT_VECTORS, binary=False)
    
    print(f"Öğrenilen anlamlı kelime sayısı: {len(w2v_model.wv.index_to_key)}")

if __name__ == "__main__":
    train_fast_embeddings()