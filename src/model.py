import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, num_classes=6, dropout=0.3, pretrained_embeddings=None, pad_idx=0):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True 
            
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),                   
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.d_model = d_model

    def forward(self, src):
        padding_mask = (src == self.pad_idx)
        
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        active_tokens = (~padding_mask).unsqueeze(-1).float() 
        
        x = x * active_tokens 
        
        sum_x = x.sum(dim=1) 
        
        counts = active_tokens.sum(dim=1)
        counts = torch.clamp(counts, min=1e-9) 
        
        x = sum_x / counts 
        
        logits = self.classifier(x)
            
        return logits


if __name__ == "__main__":
    dummy_vocab_size = 1000
    model = MiniTransformer(vocab_size=dummy_vocab_size, d_model=256, num_classes=6)
    dummy_input = torch.randint(0, dummy_vocab_size, (2, 128))
    
    dummy_input[0, 50:] = 0 
    dummy_input[1, 20:] = 0
    
    print(f"Giriş Boyutu: {dummy_input.shape}")
    try:
        output = model(dummy_input)
        print("Model testi başarılı.")
        print(f"Çıktı Boyutu: {output.shape}") 
    except Exception as e:
        print(f"HATA:\n{e}")