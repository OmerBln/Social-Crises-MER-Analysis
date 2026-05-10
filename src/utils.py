import torch
import re

MAX_SEQ_LEN = 128

EMOTION_LABELS = {
    0: 'Hüzün',
    1: 'Mutlu',
    2: 'Sevgi',
    3: 'Öfke',
    4: 'Korku',
    5: 'Şaşkınlık'
}


def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text, vocab, max_len=MAX_SEQ_LEN):
    text = clean_text(text)
    words = text.split()

    unk_idx, pad_idx = vocab["<UNK>"], vocab["<PAD>"]
    sos_idx, eos_idx = vocab["<SOS>"], vocab["<EOS>"]

    ids = [vocab.get(w, unk_idx) for w in words[:max_len - 2]]
    ids = [sos_idx] + ids + [eos_idx]

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids


def top_k_pool(stacked_logits: torch.Tensor, k: int) -> torch.Tensor:
    num_chunks = stacked_logits.shape[0]
    if num_chunks <= k:
        return stacked_logits.mean(dim=0)
    top_k_values, _ = torch.topk(stacked_logits, k, dim=0)
    return top_k_values.mean(dim=0)
