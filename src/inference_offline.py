import torch
import pandas as pd
import json
import os
import re
import glob
from easynmt import EasyNMT
from model import MiniTransformer
from utils import preprocess_text, top_k_pool, EMOTION_LABELS, MAX_SEQ_LEN

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_emotion_model.pth")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")


BATCH_SIZE = 4
SAMPLE_LIMIT = 500

WORD_LIMIT = 400
CHUNK_SIZE = 25
TOP_K = 3



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_translator = None

def get_translator():
    global _translator
    if _translator is None:
        print("--- Çeviri Motoru (EasyNMT) Yükleniyor ---")
        _translator = EasyNMT('opus-mt')
        print(f"Çeviri Motoru Hazır! (Cihaz: {device})")
    return _translator



def load_system():
    print("--- Sistem (Torch Model) Yükleniyor ---")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    model = MiniTransformer(vocab_size=len(vocab), d_model=256, num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, vocab

def translate_easynmt(text_list, source_lang):
    translations = get_translator().translate(text_list, source_lang=source_lang, target_lang='en', batch_size=BATCH_SIZE)
    return translations



def run_inference():
    model, vocab = load_system()
    raw_files = glob.glob(os.path.join(DATA_DIR, "raw", "*_raw.csv"))

    analiz_dir = os.path.join(DATA_DIR, "analiz")
    os.makedirs(analiz_dir, exist_ok=True)

    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        out_name = "analyzed_" + file_name.replace("_raw.csv", ".csv")
        out_path = os.path.join(analiz_dir, out_name)

        df = pd.read_csv(file_path).head(SAMPLE_LIMIT)
        TEXT_COLUMN = 'lyrics' if 'lyrics' in df.columns else 'text'

        is_english = file_name.upper().startswith("EN_")

        # Dil kodunu dosya adından çıkar
        lang_code = file_name.split("_")[0].lower() if "_" in file_name else "en"
    
        results = []
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i : i+BATCH_SIZE]
            original_texts = batch_df[TEXT_COLUMN].tolist()
            
            truncated_texts = [
                " ".join(str(re.sub(r'\[.*?\]', '', str(t))).split()[:WORD_LIMIT])
                for t in original_texts
            ]
            
            if is_english:
                translated_texts = truncated_texts
            else:
                translated_texts = translate_easynmt(truncated_texts, source_lang=lang_code)
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_chunks = []
            chunk_to_song_map = []
            
            for j, text in enumerate(translated_texts):
                words = str(text).split()
                chunks = [words[k : k+CHUNK_SIZE] for k in range(0, min(WORD_LIMIT, len(words)), CHUNK_SIZE)]
                
                if not chunks: 
                    chunks = [[""]]
                    
                for chunk in chunks:
                    all_chunks.append(" ".join(chunk) if isinstance(chunk, list) else chunk)
                    chunk_to_song_map.append(j)
            
            input_ids_list = [preprocess_text(chunk, vocab) for chunk in all_chunks]
            input_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(device)
            
            with torch.inference_mode():
                logits = model(input_tensor)
            
            song_logits = {j: [] for j in range(len(translated_texts))}
            
            for idx, song_idx in enumerate(chunk_to_song_map):
                song_logits[song_idx].append(logits[idx])
                
            for j, text in enumerate(original_texts):
                stacked_logits = torch.stack(song_logits[j])
                avg_logits = top_k_pool(stacked_logits, TOP_K)
                final_probs = torch.softmax(avg_logits, dim=0)
                pred_idx = torch.argmax(final_probs).item()
                
                row_data = batch_df.iloc[j]

                results.append({
                    'year': str(row_data.get('year', '')),
                    'artist': str(row_data.get('artist', '')),
                    'title': str(row_data.get('title', '')),
                    'original_snippet': str(text)[:50],
                    'translated_snippet': str(translated_texts[j])[:50],
                    'predicted_emotion': EMOTION_LABELS[pred_idx],
                    'score_sad': final_probs[0].item(),
                    'score_joy': final_probs[1].item(),
                    'score_love': final_probs[2].item(),
                    'score_anger': final_probs[3].item(),
                    'score_fear': final_probs[4].item(),
                    'score_surprise': final_probs[5].item()
                })
            
            print(f"   {min(i+BATCH_SIZE, len(df))}/{len(df)} işlendi...", end="\r")
            
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n   TAMAMLANDI: {out_name}\n")

if __name__ == "__main__":
    run_inference()