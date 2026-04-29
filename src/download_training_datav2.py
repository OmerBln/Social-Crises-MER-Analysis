import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LABEL_MAP = {
    0: 'Hüzün', 
    1: 'Mutlu', 
    2: 'Sevgi', 
    3: 'Öfke', 
    4: 'Korku', 
    5: 'Şaşkınlık'
}

def download_and_process():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("\nHATA: 'datasets' kütüphanesi eksik!")
        print("Lütfen terminale şunu yazarak yükleyin: pip install datasets")
        return
    
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    
    df_train = dataset['train'].to_pandas()
    df_val = dataset['validation'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    full_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"Toplam Ham Veri Sayısı: {len(full_df)}")
    
    processed_data = []
    for _, row in full_df.iterrows():
        processed_data.append({
            'text': row['text'],
            'label': row['label']
        })
        
    new_df = pd.DataFrame(processed_data)
    
    print("\n6 Sınıflı Yeni Dağılım:")
    for label, count in new_df['label'].value_counts().sort_index().items():
        print(f"   {label} ({LABEL_MAP[label]}): {count} örnek")
        
    save_path = os.path.join(DATA_DIR, "labeled_emotion_data.csv")
    new_df.to_csv(save_path, index=False)
    print(f"\nBAŞARILI: {save_path} kaydedildi. Toplam {len(new_df)} temiz örnek.")

if __name__ == "__main__":
    download_and_process()