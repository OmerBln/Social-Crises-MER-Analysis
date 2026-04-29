import torch
import pandas as pd
import json
import os
import re
from easynmt import EasyNMT
from model import MiniTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_PATH = "final_emotion_model.pth"
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

MAX_SEQ_LEN = 128 
BATCH_SIZE = 4
SAMPLE_LIMIT = 500

EMOTION_LABELS = {
    0: 'Hüzün',
    1: 'Mutlu',
    2: 'Sevgi',
    3: 'Öfke',
    4: 'Korku',
    5: 'Şaşkınlık'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Çeviri Motoru (EasyNMT) Yükleniyor ---")
translator = EasyNMT('opus-mt')
print(f"Çeviri Motoru Hazır! (Cihaz: {device})")

def preprocess_text(text, vocab, max_len=MAX_SEQ_LEN):
    if not isinstance(text, str): text = ""
    text = text.lower()
    
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split() 
    
    unk_idx, pad_idx = vocab["<UNK>"], vocab["<PAD>"]
    sos_idx, eos_idx = vocab["<SOS>"], vocab["<EOS>"]
    
    ids = [vocab.get(w, unk_idx) for w in words[:max_len - 2]]
    ids = [sos_idx] + ids + [eos_idx]
    
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
        
    return ids

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
    translations = translator.translate(text_list, source_lang=source_lang, target_lang='en', batch_size=BATCH_SIZE)
    return translations

def run_inference():
    model, vocab = load_system()
    
    files_to_process = [
        ("data_tr.csv", "tr"),
        ("data_en.csv", "en"),
        ("data_ja.csv", "ja"),
        ("data_fr.csv", "fr"),
        ("data_es.csv", "es")
    ]
    
    for file_name, lang_code in files_to_process:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Uyarı: {file_name} bulunamadı, atlanıyor.")
            continue
            
        print(f"\n-> Dosya İşleniyor: {file_name} (Dil: {lang_code})")
        df = pd.read_csv(file_path)
        
        if 'lyrics' in df.columns:
            TEXT_COLUMN = 'lyrics'
        elif 'text' in df.columns:
            TEXT_COLUMN = 'text'
        else:
            print(f"   Hata: {file_name} içinde geçerli bir metin sütunu bulunamadı!")
            continue
            
        df = df.head(SAMPLE_LIMIT)
        print(f"   Hedef: {len(df)} şarkı analiz edilecek.")
        
        results = []
        
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i : i+BATCH_SIZE]
            original_texts = batch_df[TEXT_COLUMN].tolist()
            
            truncated_texts = []
            for text in original_texts:
                clean_text = re.sub(r'\[.*?\]', '', str(text))
                words = str(text).split()[:75]
                truncated_texts.append(" ".join(words))
            
            if lang_code != 'en':
                translated_texts = translate_easynmt(truncated_texts, source_lang=lang_code)
            else:
                translated_texts = truncated_texts
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_chunks = []
            chunk_to_song_map = []
            
            for j, text in enumerate(translated_texts):
                words = str(text).split()
                chunks = [words[k : k+25] for k in range(0, min(75, len(words)), 25)]
                
                if not chunks: 
                    chunks = [[""]]
                    
                for chunk in chunks:
                    if isinstance(chunk, list):
                        chunk_str = " ".join(chunk)
                    else:
                        chunk_str = chunk
                    all_chunks.append(chunk_str)
                    chunk_to_song_map.append(j)
            
            input_ids_list = [preprocess_text(chunk, vocab) for chunk in all_chunks]
            input_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
            
            song_probs = {j: [] for j in range(len(translated_texts))}
            
            for idx, song_idx in enumerate(chunk_to_song_map):
                song_probs[song_idx].append(probs[idx])
                
            for j, text in enumerate(original_texts):
                stacked_probs = torch.stack(song_probs[j]) 
                avg_probs = stacked_probs.mean(dim=0) 
                
                pred_idx = torch.argmax(avg_probs).item()
                
                results.append({
                    'original_snippet': str(text)[:50],
                    'translated_snippet': str(translated_texts[j])[:50] if lang_code != 'en' else "-",
                    'predicted_emotion': EMOTION_LABELS[pred_idx],
                    'score_sad': avg_probs[0].item(),
                    'score_joy': avg_probs[1].item(),
                    'score_love': avg_probs[2].item(),
                    'score_anger': avg_probs[3].item(),
                    'score_fear': avg_probs[4].item(),
                    'score_surprise': avg_probs[5].item()
                })
            
            print(f"   {min(i+BATCH_SIZE, len(df))}/{len(df)} işlendi...", end="\r")
            
        out_df = pd.DataFrame(results)
        out_path = os.path.join(DATA_DIR, f"analyzed_{lang_code}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n   TAMAMLANDI: analyzed_{lang_code}.csv\n")

if __name__ == "__main__":
    run_inference()