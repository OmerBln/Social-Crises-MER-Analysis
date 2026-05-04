import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import glob
import os
import sys
import re
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ANALIZ_DIR = os.path.join(DATA_DIR, "analiz")
TREND_DIR  = os.path.join(DATA_DIR, "trend")
os.makedirs(TREND_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

SCORE_COLS = ['score_sad', 'score_joy', 'score_love',
              'score_anger', 'score_fear', 'score_surprise']
EMOTION_LABELS_TR = {
    'score_sad':      'Huzun',
    'score_joy':      'Mutlu',
    'score_love':     'Sevgi',
    'score_anger':    'Ofke',
    'score_fear':     'Korku',
    'score_surprise': 'Saskinlik',
}
EMOTION_LABELS_SHORT = {
    'score_sad':      'Sad',
    'score_joy':      'Joy',
    'score_love':     'Love',
    'score_anger':    'Anger',
    'score_fear':     'Fear',
    'score_surprise': 'Surprise',
}
EMOTION_COLORS = {
    'score_sad':      '#5B8DEE',
    'score_joy':      '#F5A623',
    'score_love':     '#E74C7A',
    'score_anger':    '#E74C3C',
    'score_fear':     '#8E44AD',
    'score_surprise': '#2ECC71',
}


def format_event_label(name):
    parts = name.split('_', 2)
    if len(parts) >= 3:
        label = re.sub(r'([a-z])([A-Z])', r'\1 \2', parts[2])
        return f"{parts[0]} {parts[1]} {label}"
    return name


def extract_event_years(event_name):
    parts = event_name.split('_')
    for p in parts:
        if p.isdigit() and len(p) == 4:
            return int(p)
    return None


def discover_event_pairs():
    event_files = glob.glob(os.path.join(ANALIZ_DIR, "analyzed_*.csv"))

    event_map = {}
    baseline_map = {}

    for path in event_files:
        fname = os.path.basename(path).replace("analyzed_", "").replace(".csv", "")
        if fname.endswith("_BASELINE"):
            base_name = fname.replace("_BASELINE", "")
            baseline_map[base_name] = path
        else:
            event_map[fname] = path

    pairs = []
    for event_name, event_path in sorted(event_map.items()):
        baseline_path = baseline_map.get(event_name)
        if baseline_path:
            pairs.append({
                "event_name":    event_name,
                "event_path":    event_path,
                "baseline_path": baseline_path,
            })
        else:
            print(f"  {event_name} icin Baseline dosyasi bulunamadi, atlanacak.")

    return pairs


def build_yearly_timeline(event_df, baseline_df, event_name):
    combined = pd.concat([baseline_df, event_df], ignore_index=True)

    if 'year' not in combined.columns:
        print(f"  'year' sutunu bulunamadi. inference_online.py yeniden calistirilmali.")
        return None

    combined['year'] = combined['year'].astype(str).str.extract(r'(\d{4})')[0]
    combined = combined.dropna(subset=['year'])
    combined['year'] = combined['year'].astype(int)

    existing_scores = [c for c in SCORE_COLS if c in combined.columns]
    if not existing_scores:
        return None

    yearly = combined.groupby('year')[existing_scores].agg(['mean', 'count', 'std'])

    yearly.columns = ['_'.join(col) for col in yearly.columns]

    yearly = yearly.sort_index()

    event_start_year = extract_event_years(event_name)

    yearly['is_event'] = yearly.index >= event_start_year if event_start_year else False

    return yearly


def plot_yearly_emotion_lines(event_name, yearly_df, event_start_year):
    if yearly_df is None or yearly_df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    years = yearly_df.index.values

    for idx, col in enumerate(SCORE_COLS):
        ax = axes[idx]
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"

        if mean_col not in yearly_df.columns:
            continue

        means = yearly_df[mean_col].values
        stds  = yearly_df[std_col].values if std_col in yearly_df.columns else np.zeros_like(means)

        baseline_mask = ~yearly_df['is_event'].values
        event_mask    = yearly_df['is_event'].values

        if baseline_mask.any():
            ax.plot(years[baseline_mask], means[baseline_mask],
                    'o-', color=EMOTION_COLORS[col], linewidth=2, markersize=7,
                    label='Baseline', zorder=3)
            ax.fill_between(years[baseline_mask],
                            means[baseline_mask] - stds[baseline_mask],
                            means[baseline_mask] + stds[baseline_mask],
                            alpha=0.15, color=EMOTION_COLORS[col])

        if event_mask.any():
            ax.plot(years[event_mask], means[event_mask],
                    's-', color='#E74C3C', linewidth=2.5, markersize=9,
                    label='Olay Donemi', zorder=4)
            ax.fill_between(years[event_mask],
                            means[event_mask] - stds[event_mask],
                            means[event_mask] + stds[event_mask],
                            alpha=0.15, color='#E74C3C')

        if baseline_mask.any() and event_mask.any():
            bridge_years = [years[baseline_mask][-1], years[event_mask][0]]
            bridge_means = [means[baseline_mask][-1], means[event_mask][0]]
            ax.plot(bridge_years, bridge_means, '--', color='gray',
                    linewidth=1, alpha=0.5)

        if event_start_year:
            ax.axvline(x=event_start_year - 0.5, color='red', linestyle='--',
                       linewidth=1.2, alpha=0.6, label='Olay Baslangici')

        ax.set_title(f"{EMOTION_LABELS_SHORT[col]} ({EMOTION_LABELS_TR[col]})",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Yil', fontsize=10)
        ax.set_ylabel('Ortalama Skor', fontsize=10)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45, fontsize=8)
        ax.legend(fontsize=8, loc='best')

        for y_val, x_val in zip(means, years):
            ax.annotate(f'{y_val:.3f}', (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=7, color='#333')

    fig.suptitle(f'{format_event_label(event_name)}\nYil Bazli Duygu Trendi',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(TREND_DIR, f"trend_yearly_{event_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Grafik {save_path}")


def plot_combined_trend_line(event_name, yearly_df, event_start_year):
    if yearly_df is None or yearly_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    years = yearly_df.index.values

    for col in SCORE_COLS:
        mean_col = f"{col}_mean"
        if mean_col not in yearly_df.columns:
            continue
        means = yearly_df[mean_col].values
        ax.plot(years, means, 'o-', color=EMOTION_COLORS[col], linewidth=2,
                markersize=6, label=f"{EMOTION_LABELS_SHORT[col]}")

    if event_start_year:
        ax.axvline(x=event_start_year - 0.5, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Olay Baslangici')
        ax.axvspan(event_start_year - 0.5, years[-1] + 0.5,
                   alpha=0.08, color='red')

    ax.set_title(f'{format_event_label(event_name)} - Tum Duygular',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Yil', fontsize=12)
    ax.set_ylabel('Ortalama Duygu Skoru', fontsize=12)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.legend(fontsize=10, loc='best', ncols=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(TREND_DIR, f"trend_combined_{event_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Grafik {save_path}")


def run_welch_ttest(event_df, baseline_df):
    results = {}
    for col in SCORE_COLS:
        if col not in event_df.columns or col not in baseline_df.columns:
            continue
        ev_vals = event_df[col].dropna()
        bl_vals = baseline_df[col].dropna()
        if len(ev_vals) < 2 or len(bl_vals) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(ev_vals, bl_vals, equal_var=False)

        ev_mean = ev_vals.mean()
        bl_mean = bl_vals.mean()
        delta   = ev_mean - bl_mean

        pooled_std = np.sqrt(ev_vals.std()**2 / 2 + bl_vals.std()**2 / 2)
        cohens_d = delta / pooled_std if pooled_std > 1e-9 else 0.0

        results[col] = {
            'event_mean':    ev_mean,
            'baseline_mean': bl_mean,
            'delta':         delta,
            'delta_pct':     (delta / bl_mean * 100) if bl_mean > 1e-9 else 0.0,
            't_stat':        t_stat,
            'p_value':       p_value,
            'cohens_d':      cohens_d,
            'significant':   p_value < 0.05,
        }
    return results


def plot_delta_heatmap(all_results):
    event_names = []
    delta_matrix = []

    for event_name, test_results in all_results.items():
        row = []
        for col in SCORE_COLS:
            if col in test_results:
                row.append(test_results[col]['delta'])
            else:
                row.append(0.0)
        delta_matrix.append(row)
        event_names.append(format_event_label(event_name))

    if not delta_matrix:
        return

    df_heat = pd.DataFrame(
        delta_matrix,
        index=event_names,
        columns=[EMOTION_LABELS_SHORT[c] for c in SCORE_COLS]
    )

    fig, ax = plt.subplots(figsize=(14, max(8, len(event_names) * 0.5)))
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap='RdBu_r', center=0,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Delta (Olay - Baseline)'})
    ax.set_title('Tum Olaylarin Duygu Degisim Haritasi\n(Pozitif = Olayda Artti, Negatif = Olayda Azaldi)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    save_path = os.path.join(TREND_DIR, "trend_delta_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Heatmap {save_path}")


def plot_significance_summary(all_results):
    event_names = []
    sig_matrix = []

    for event_name, test_results in all_results.items():
        row = []
        for col in SCORE_COLS:
            if col in test_results:
                r = test_results[col]
                if r['significant'] and r['delta'] > 0:
                    row.append(1)
                elif r['significant'] and r['delta'] < 0:
                    row.append(-1)
                else:
                    row.append(0)
            else:
                row.append(0)
        sig_matrix.append(row)
        event_names.append(format_event_label(event_name))

    if not sig_matrix:
        return

    df_sig = pd.DataFrame(
        sig_matrix,
        index=event_names,
        columns=[EMOTION_LABELS_SHORT[c] for c in SCORE_COLS]
    )

    cmap = matplotlib.colors.ListedColormap(['#2ECC71', '#F0F0F0', '#E74C3C'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, max(8, len(event_names) * 0.5)))
    sns.heatmap(df_sig, annot=False, cmap=cmap, linewidths=1, ax=ax,
                cbar=False, vmin=-1.5, vmax=1.5)

    for i in range(len(event_names)):
        for j in range(len(SCORE_COLS)):
            val = sig_matrix[i][j]
            if val == 1:
                ax.text(j + 0.5, i + 0.5, '▲', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')
            elif val == -1:
                ax.text(j + 0.5, i + 0.5, '▼', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')
            else:
                ax.text(j + 0.5, i + 0.5, '─', ha='center', va='center',
                        fontsize=12, color='#999')

    ax.set_title('Istatistiksel Anlamlilik Ozeti (p < 0.05)\n'
                 '▲ Kirmizi = Olayda Anlamli Artis  |  ▼ Yesil = Olayda Anlamli Azalis  |  ─ Gri = Anlamli Degisim Yok',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    save_path = os.path.join(TREND_DIR, "trend_significance_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Anlamlilik {save_path}")


def interpret_result(r):
    if not r['significant']:
        return "Anlamli degisim yok: Bu duygu olay oncesi donemle benzer seyrediyor."

    direction = "artti" if r['delta'] > 0 else "azaldi"

    abs_d = abs(r['cohens_d'])
    if abs_d >= 0.8:
        effect = "buyuk"
    elif abs_d >= 0.5:
        effect = "orta"
    else:
        effect = "kucuk"

    return (f"Olay doneminde {direction} ({effect} etki). "
            f"Delta: {r['delta']:+.3f} ({r['delta_pct']:+.1f}%). "
            f"Bu duygu olaydan kaynaklanmis olabilir.")


def generate_csv_report(all_results, all_yearly_data):
    rows = []

    for event_name, test_results in all_results.items():
        for col in SCORE_COLS:
            if col not in test_results:
                continue
            r = test_results[col]
            rows.append({
                'event':         event_name,
                'year':          'TOPLAM',
                'emotion':       EMOTION_LABELS_SHORT[col],
                'emotion_tr':    EMOTION_LABELS_TR[col],
                'baseline_mean': round(r['baseline_mean'], 4),
                'event_mean':    round(r['event_mean'], 4),
                'delta':         round(r['delta'], 4),
                'delta_pct':     round(r['delta_pct'], 2),
                't_stat':        round(r['t_stat'], 4),
                'p_value':       round(r['p_value'], 6),
                'cohens_d':      round(r['cohens_d'], 4),
                'significant':   r['significant'],
                'interpretation': interpret_result(r),
            })

    for event_name, yearly_df in all_yearly_data.items():
        if yearly_df is None:
            continue
        for year_val in yearly_df.index:
            for col in SCORE_COLS:
                mean_col = f"{col}_mean"
                count_col = f"{col}_count"
                if mean_col not in yearly_df.columns:
                    continue
                is_event = yearly_df.loc[year_val, 'is_event']
                rows.append({
                    'event':         event_name,
                    'year':          str(year_val),
                    'emotion':       EMOTION_LABELS_SHORT[col],
                    'emotion_tr':    EMOTION_LABELS_TR[col],
                    'baseline_mean': '',
                    'event_mean':    '',
                    'delta':         '',
                    'delta_pct':     '',
                    'yearly_mean':   round(yearly_df.loc[year_val, mean_col], 4),
                    'yearly_count':  int(yearly_df.loc[year_val, count_col]) if count_col in yearly_df.columns else '',
                    'period':        'OLAY' if is_event else 'BASELINE',
                    't_stat':        '',
                    'p_value':       '',
                    'cohens_d':      '',
                    'significant':   '',
                    'interpretation': '',
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(TREND_DIR, "trend_analysis_report.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  RAPOR {csv_path}")
    return df


def print_summary_table(report_df):
    print("\n" + "=" * 90)
    print("  Trend Analizi Ozeti")
    print("=" * 90)

    totals = report_df[report_df['year'] == 'TOPLAM']
    sig_df = totals[totals['significant'] == True]

    if sig_df.empty:
        print("  Hicbir olayda istatistiksel olarak anlamli bir duygu degisimi tespit edilemedi.")
        print("  Bu, incelenen olaylarin muzikteki duygusal ifadeyi belirgin olarak degistirmedigini gosterir.")
        return

    for event_name in sig_df['event'].unique():
        ev_sig = sig_df[sig_df['event'] == event_name]
        print(f"\n  {format_event_label(event_name)}")
        print(f"  {'-' * 70}")
        for _, row in ev_sig.iterrows():
            arrow = "+" if row['delta'] > 0 else "-"
            print(f"    {arrow} {row['emotion_tr']:>10} ({row['emotion']:>8}): "
                  f"Baseline={row['baseline_mean']:.3f} -> Olay={row['event_mean']:.3f} "
                  f"(Delta={row['delta']:+.3f}, p={row['p_value']:.4f})")

    print("\n" + "=" * 90)

    total_tests = len(totals)
    sig_count   = len(sig_df)
    print(f"  Toplam {total_tests} test yapildi, {sig_count} tanesi anlamli (p < 0.05).")
    if total_tests > 0:
        print(f"  Anlamlilik orani: %{sig_count / total_tests * 100:.1f}")


def run_trend_analysis():
    print("\n" + "=" * 62)
    print("  Trend Analizi")
    print("=" * 62)

    pairs = discover_event_pairs()

    if not pairs:
        print("\n   Hicbir olay-baseline cifti bulunamadi")
        print("   once fetch_historical_datav8.py ve inference_online.py")
        return

    print(f"\n  {len(pairs)} adet olay-baseline cifti bulundu.\n")

    all_results = {}
    all_yearly_data = {}

    for pair in pairs:
        event_name = pair["event_name"]
        event_start_year = extract_event_years(event_name)

        print(f"\n  {'_' * 50}")
        print(f"  {format_event_label(event_name)}")
        print(f"  {'_' * 50}")

        try:
            event_df    = pd.read_csv(pair["event_path"])
            baseline_df = pd.read_csv(pair["baseline_path"])
        except Exception as e:
            print(f"  Dosya okunamadi: {e}")
            continue

        print(f"  Olay donemi:   {len(event_df)} sarki")
        print(f"  Baseline:      {len(baseline_df)} sarki")

        test_results = run_welch_ttest(event_df, baseline_df)
        if test_results:
            all_results[event_name] = test_results

        yearly_df = build_yearly_timeline(event_df, baseline_df, event_name)
        if yearly_df is not None and not yearly_df.empty:
            all_yearly_data[event_name] = yearly_df
            plot_yearly_emotion_lines(event_name, yearly_df, event_start_year)
            plot_combined_trend_line(event_name, yearly_df, event_start_year)
            print(f"  Yillar: {list(yearly_df.index)}")
        else:
            print(f"  Yil bazli veri olusturulamadi.")

    if not all_results and not all_yearly_data:
        print("\n   Hicbir olay icin analiz yapilamadi!")
        return

    if all_results:
        plot_delta_heatmap(all_results)
        plot_significance_summary(all_results)

    report_df = generate_csv_report(all_results, all_yearly_data)
    print_summary_table(report_df)

    print(f"\n  Tum trend analiz ciktilari: {TREND_DIR}")
    print("  Trend analizi tamamlandi!")


if __name__ == "__main__":
    run_trend_analysis()
