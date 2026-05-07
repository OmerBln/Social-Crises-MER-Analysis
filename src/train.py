import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import numpy as np
from model import MiniTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_emotion_model.pth")
VECTOR_PATH = os.path.join(DATA_DIR, "lyrics_vectors.txt")
BATCH_SIZE = 32
EPOCHS = 15 
LEARNING_RATE = 1e-4
D_MODEL = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():  
    train_x, train_y = torch.load(os.path.join(DATA_DIR, "real_train.pt"), weights_only=True)
    val_x, val_y = torch.load(os.path.join(DATA_DIR, "real_val.pt"), weights_only=True)
    return train_x, train_y, val_x, val_y

def load_pretrained_vectors(vocab, vector_path, d_model):
    print("Word2Vec Yükleniyor ")
    
    embeddings = torch.randn(len(vocab), d_model)
    
    embeddings[vocab["<PAD>"]] = torch.zeros(d_model)
    
    found_words = 0
    with open(vector_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().split()
        if len(first_line) != 2:
            f.seek(0) 
            
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                embeddings[idx] = vec
                found_words += 1
                
    print(f"Sözlükteki {len(vocab)} kelimenin {found_words} tanesi Word2Vec ile başarıyla eşleşti!\n")
    return embeddings

def train():
    with open(os.path.join(DATA_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Sözlük Boyutu: {vocab_size}")
    print(f"Cihaz: {device}")

    pretrained_weights = load_pretrained_vectors(vocab, VECTOR_PATH, D_MODEL)

    model = MiniTransformer(vocab_size=vocab_size, d_model=D_MODEL, num_classes=6, pretrained_embeddings=pretrained_weights).to(device)
    
    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    other_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': LEARNING_RATE},
        {'params': embedding_params, 'lr': LEARNING_RATE}
    ], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    train_x, train_y, val_x, val_y = load_data() 
    
    labels = train_y.cpu().numpy()
    class_counts = np.bincount(labels, minlength=6)
    class_weights = len(labels) / (6.0 * (class_counts + 1e-5)) 
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    best_val_loss = float('inf') 
    patience_counter = 0
    PATIENCE = 5

    for epoch in range(EPOCHS):
        
        if epoch == 0:
            print("Embedding katmanı donduruldu.")
            model.embedding.weight.requires_grad = False
        elif epoch == 3:
            print("Embedding katmanı çözüldü, fine-tuning başlıyor.")
            model.embedding.weight.requires_grad = True
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
            for p in embedding_params:
                if p in optimizer.state:
                    del optimizer.state[p]

        model.train()
        total_loss = 0
        
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} Özet -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Doğruluk: {accuracy:.2f}% | LR: {current_lr}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"En iyi model kaydedildi. (Val Loss: {best_val_loss:.4f})\n")
        else:
            patience_counter += 1
            print(f"Model iyileşmedi. ({patience_counter}/{PATIENCE}) (En İyi Val Loss: {best_val_loss:.4f})\n")
            if patience_counter >= PATIENCE:
                print("Early stopping tetiklendi!")
                break

if __name__ == "__main__":
    train()