import torch
import pandas as pd
import json
import os
import re
import glob
import time
from deep_translator import GoogleTranslator
from model import MiniTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_emotion_model.pth")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

MAX_SEQ_LEN = 128
BATCH_SIZE = 6
SAMPLE_LIMIT = 500

WORD_LIMIT = 400
CHUNK_SIZE = 25

TOP_K = 3

EMOTION_LABELS = {
    0: 'Hüzün',
    1: 'Mutlu',
    2: 'Sevgi',
    3: 'Öfke',
    4: 'Korku',
    5: 'Şaşkınlık'
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_text(text, vocab, max_len=MAX_SEQ_LEN):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    unk_idx, pad_idx, sos_idx, eos_idx = (
        vocab["<UNK>"], vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"]
    )

    ids = [vocab.get(w, unk_idx) for w in words[:max_len - 2]]
    ids = [sos_idx] + ids + [eos_idx]

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids


def load_system():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    model = MiniTransformer(vocab_size=len(vocab), d_model=256, num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, vocab


_global_translator = None

def get_translator():
    global _global_translator
    if _global_translator is None:
        _global_translator = GoogleTranslator(source='auto', target='en')
    return _global_translator

TRANSLATION_CACHE_FILE = os.path.join(DATA_DIR, "translation_cache.json")
translation_cache = {}

if os.path.exists(TRANSLATION_CACHE_FILE):
    try:
        with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
            translation_cache = json.load(f)
    except Exception:
        pass

def save_translation_cache():
    try:
        with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def translate_deep(text_list):
    translations = []
    new_translations = False
    
    for text in text_list:
        if text in translation_cache:
            translations.append(translation_cache[text])
            continue
            
        try:
            res = get_translator().translate(text)
            final_res = res if res else text
            translation_cache[text] = final_res
            translations.append(final_res)
            new_translations = True
            time.sleep(0.5)
        except Exception:
            translation_cache[text] = text
            translations.append(text)
            new_translations = True
            time.sleep(1)
            
    if new_translations:
        save_translation_cache()
        
    return translations


def top_k_pool(stacked_logits: torch.Tensor, k: int) -> torch.Tensor:
    num_chunks = stacked_logits.shape[0]

    if num_chunks <= k:
        return stacked_logits.mean(dim=0)

    top_k_values, _ = torch.topk(stacked_logits, k, dim=0)
    return top_k_values.mean(dim=0)


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
        
        results = []
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i: i + BATCH_SIZE]
            original_texts = batch_df[TEXT_COLUMN].tolist()

            truncated_texts = [
                " ".join(str(re.sub(r'\[.*?\]', '', str(t))).split()[:WORD_LIMIT])
                for t in original_texts
            ]
            
            if is_english:
                translated_texts = truncated_texts
            else:
                translated_texts = translate_deep(truncated_texts)

            all_chunks = []
            chunk_to_song_map = []

            for j, text in enumerate(translated_texts):
                words = str(text).split()
                chunks = [
                    words[k: k + CHUNK_SIZE]
                    for k in range(0, min(WORD_LIMIT, len(words)), CHUNK_SIZE)
                ]

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

                row_idx = i + j
                row_data = df.iloc[row_idx] if row_idx < len(df) else {}

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

            print(f"   {min(i + BATCH_SIZE, len(df))}/{len(df)} işlendi...", end="\r")

        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n   TAMAMLANDI: {out_name}\n")


if __name__ == "__main__":
    run_inference()