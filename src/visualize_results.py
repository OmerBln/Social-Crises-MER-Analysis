import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

import re

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

def format_event_label(name):
    parts = name.split('_', 2)
    if len(parts) >= 3:
        label = re.sub(r'([a-z])([A-Z])', r'\1 \2', parts[2])
        return f"{parts[0]} {parts[1]}\n{label}"
    return name

def load_all_analyzed_data():
    analiz_dir = os.path.join(DATA_DIR, "analiz")
    file_paths = [p for p in glob.glob(os.path.join(analiz_dir, "analyzed_*.csv"))
                  if "_BASELINE" not in os.path.basename(p)]
    
    if not file_paths:
        print(f"HATA: 'analyzed_' ile başlayan sonuç dosyası bulunamadı!")
        print(f"Aranan dizin: {analiz_dir}")
        print("Lütfen grafikleri çizdirmeden önce 'inference_online.py' dosyasını çalıştırın.")
        return None

    all_data = []
    
    for path in file_paths:
        file_name = os.path.basename(path)
        
        event_name = file_name.replace("analyzed_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(path)
            df['Event'] = event_name
            all_data.append(df)
        except Exception as e:
            print(f"Dosya okunurken hata oluştu ({file_name}): {e}")
            
    if not all_data:
        return None
        
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Toplam {len(full_df)} adet analiz edilmiş şarkı parçası başarıyla yüklendi.")
    return full_df

def plot_emotion_distribution_by_event(df):
    plt.figure(figsize=(18, 9))
    
    emotion_counts = df.groupby(['Event', 'predicted_emotion']).size().reset_index(name='Count')
    
    sns.barplot(data=emotion_counts, x='Event', y='Count', hue='predicted_emotion', palette='Set2')
    
    plt.title('Toplumsal Krizlere Göre Şarkılardaki Baskın Duygu Dağılımı', fontsize=16, fontweight='bold')
    plt.xlabel('Krizler / Olaylar', fontsize=12)
    plt.ylabel('Şarkı Parçası Sayısı', fontsize=12)
    
    ax = plt.gca()
    fig = plt.gcf()
    fig.canvas.draw()
    labels = [format_event_label(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = os.path.join(PLOTS_DIR, "1_Krizlere_Gore_Duygu_Dagilimi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {save_path}")

def plot_average_emotion_scores(df):
    plt.figure(figsize=(12, 8))
    
    score_cols = ['score_sad', 'score_joy', 'score_love', 'score_anger', 'score_fear', 'score_surprise']
    
    existing_score_cols = [col for col in score_cols if col in df.columns]
    
    if not existing_score_cols:
        print("Hata: Skor sütunları bulunamadı. inference_online.py çıktısını kontrol edin.")
        return
        
    avg_scores = df.groupby('Event')[existing_score_cols].mean()
    
    avg_scores.columns = [col.replace('score_', '').capitalize() for col in avg_scores.columns]
    
    sns.heatmap(avg_scores, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=.5)
    
    plt.title('Toplumsal Krizlerin Müzikteki Ortalama Duygu Şiddeti (0-1 Arası)', fontsize=16, fontweight='bold')
    plt.xlabel('Duygular', fontsize=12)
    plt.ylabel('Krizler / Olaylar', fontsize=12)
    
    save_path = os.path.join(PLOTS_DIR, "2_Ortalama_Duygu_Siddeti_Isi_Haritasi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {save_path}")

def plot_overall_emotion_pie(df):
    plt.figure(figsize=(10, 8))
    
    emotion_counts = df['predicted_emotion'].value_counts()
    
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("pastel"), wedgeprops={'edgecolor': 'white'})
    
    plt.title('Tüm Kriz Dönemlerindeki Genel Duygu Dağılımı', fontsize=16, fontweight='bold')
    
    save_path = os.path.join(PLOTS_DIR, "3_Genel_Duygu_Pasta_Grafigi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {save_path}")

def plot_emotion_by_language(df):
    lang_map = {'TR': 'Türkçe', 'EN': 'English', 'JA': 'Japanese', 'FR': 'Français', 'ES': 'Español'}
    
    df_copy = df.copy()
    df_copy['Language'] = df_copy['Event'].str.split('_').str[0].map(lang_map).fillna('Diğer')
    
    if df_copy['Language'].nunique() < 2:
        print("Dil bazında karşılaştırma için en az 2 dil gerekli, atlanıyor.")
        return
    
    plt.figure(figsize=(12, 7))
    
    lang_emotion = df_copy.groupby(['Language', 'predicted_emotion']).size().reset_index(name='Count')
    lang_totals = df_copy.groupby('Language').size().reset_index(name='Total')
    lang_emotion = lang_emotion.merge(lang_totals, on='Language')
    lang_emotion['Percentage'] = (lang_emotion['Count'] / lang_emotion['Total']) * 100
    
    sns.barplot(data=lang_emotion, x='Language', y='Percentage', hue='predicted_emotion', palette='Set2')
    
    plt.title('Dil Bazlı Duygu Dağılımı Karşılaştırması (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Dil', fontsize=12)
    plt.ylabel('Yüzde (%)', fontsize=12)
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = os.path.join(PLOTS_DIR, "4_Dil_Bazli_Duygu_Karsilastirmasi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {save_path}")

def main():
    df = load_all_analyzed_data()
    
    if df is not None:
        plot_emotion_distribution_by_event(df)
        plot_average_emotion_scores(df)
        plot_overall_emotion_pie(df)
        plot_emotion_by_language(df)
        print("\nTÜM İŞLEMLER TAMAM!")

if __name__ == "__main__":
    main()