from __future__ import annotations

import pandas as pd
import os, re, random, sys, time, unicodedata
from langdetect import detect
import musicbrainzngs
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

lrclib_session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
lrclib_session.mount('https://', HTTPAdapter(max_retries=retries))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

musicbrainzngs.set_useragent("MusicEmotionAnalysis", "1.0", "academic-nlp-project")

_artist_country_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "artist_country_cache.json")
_artist_country_cache: dict[str, str | None] = {}

def load_artist_country_cache():
    global _artist_country_cache
    if os.path.exists(_artist_country_cache_file):
        try:
            with open(_artist_country_cache_file, "r", encoding="utf-8") as f:
                _artist_country_cache = json.load(f)
            print(f"[Cache] {_artist_country_cache_file} yuklendi ({len(_artist_country_cache)} sanatci).")
        except Exception as e:
            print(f"[Cache] Yuklenirken hata: {e}")
            _artist_country_cache = {}

def save_artist_country_cache():
    try:
        with open(_artist_country_cache_file, "w", encoding="utf-8") as f:
            json.dump(_artist_country_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_artist_country(artist_name: str) -> str | None:
    norm_name = artist_name.lower().strip()
    if norm_name in _artist_country_cache:
        return _artist_country_cache[norm_name]
    
    print(f"      [MB] Koken: {artist_name[:20]:<20}", end=" ", flush=True)
    
    clean_name = artist_name
    for splitter in [" feat.", " ft.", " feat ", " ft ", "&", ","]:
        idx = clean_name.lower().find(splitter)
        if idx != -1:
            clean_name = clean_name[:idx].strip()
            
    safe_name = clean_name.replace('"', '').replace('\\', '')
    
    def search_country(q: str):
        try:
            res = musicbrainzngs.search_artists(query=q, limit=5)
            for art in res.get("artist-list", []):
                if "country" in art and art["country"]:
                    return art["country"]
                area = art.get("area", {})
                codes = area.get("iso-3166-1-codes", [])
                if codes and codes[0]:
                    return codes[0]
        except Exception:
            pass
        return None

    country = search_country(f'"{safe_name}"')
    
    if not country:
        time.sleep(1.1)
        country = search_country(safe_name)
        
    time.sleep(1.1)
    
    _artist_country_cache[norm_name] = country
    save_artist_country_cache()
    print(f"-> {country}" if country else "-> None")
    return country

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TARGET_EVENTS = [
    {"event_name": "TR_1999_GolcukDepremi",           "year": "1999-2000", "lang": "tr", "mb_country": "TR", "limit": 200},
    {"event_name": "TR_2001_EkonomikKriz",             "year": "2001-2002", "lang": "tr", "mb_country": "TR", "limit": 200},
    {"event_name": "TR_2013_GeziParkiOlaylari",        "year": "2013-2014", "lang": "tr", "mb_country": "TR", "limit": 200},
    {"event_name": "TR_2016_DarbeGirisimi",            "year": "2016-2017", "lang": "tr", "mb_country": "TR", "limit": 200},
    {"event_name": "TR_2023_KahramanmarasDepremleri",  "year": "2023-2024", "lang": "tr", "mb_country": "TR", "limit": 200},

    {"event_name": "EN_2001_911Saldirilari",           "year": "2001-2002", "lang": "en", "mb_country": "US", "limit": 200},
    {"event_name": "EN_2005_KatrinaKasirgasi",         "year": "2005-2006", "lang": "en", "mb_country": "US", "limit": 200},
    {"event_name": "EN_2008_KureselEkonomikKriz",      "year": "2008-2009", "lang": "en", "mb_country": "US", "limit": 200},
    {"event_name": "EN_2020_GeorgeFloydProtestolari",  "year": "2020-2021", "lang": "en", "mb_country": "US", "limit": 200},
    {"event_name": "EN_2021_KongreBinasiBaskini",      "year": "2021-2022", "lang": "en", "mb_country": "US", "limit": 200},

    {"event_name": "JA_2004_ChuetsuDepremi",           "year": "2004-2005", "lang": "ja", "mb_country": "JP", "limit": 200},
    {"event_name": "JA_2011_FukushimaTsunami",         "year": "2011-2012", "lang": "ja", "mb_country": "JP", "limit": 200},
    {"event_name": "JA_2019_ReiwaDonemineGecis",       "year": "2019-2020", "lang": "ja", "mb_country": "JP", "limit": 200},
    {"event_name": "JA_2022_ShinzoAbeSuikasti",        "year": "2022-2023", "lang": "ja", "mb_country": "JP", "limit": 200},
    {"event_name": "JA_2024_NotoYarimadasiDepremi",    "year": "2024-2025", "lang": "ja", "mb_country": "JP", "limit": 200},

    {"event_name": "FR_1998_DunyaKupasiSampiyonlugu",  "year": "1998-1999", "lang": "fr", "mb_country": "FR", "limit": 200},
    {"event_name": "FR_2005_BanliyoAyaklanmalari",     "year": "2005-2006", "lang": "fr", "mb_country": "FR", "limit": 200},
    {"event_name": "FR_2015_BataclanSaldirilari",      "year": "2015-2016", "lang": "fr", "mb_country": "FR", "limit": 200},
    {"event_name": "FR_2018_SariYeleklilerProtestosu", "year": "2018-2019", "lang": "fr", "mb_country": "FR", "limit": 200},
    {"event_name": "FR_2019_NotreDameYangini",         "year": "2019-2020", "lang": "fr", "mb_country": "FR", "limit": 200},

    {"event_name": "ES_2002_PrestigePetrolSizintisi",  "year": "2002-2003", "lang": "es", "mb_country": "ES", "limit": 200},
    {"event_name": "ES_2004_MadridTrenSaldirilari",    "year": "2004-2005", "lang": "es", "mb_country": "ES", "limit": 200},
    {"event_name": "ES_2008_EkonomikKriz",             "year": "2008-2009", "lang": "es", "mb_country": "ES", "limit": 200},
    {"event_name": "ES_2011_IndignadosHareketi",       "year": "2011-2012", "lang": "es", "mb_country": "ES", "limit": 200},
    {"event_name": "ES_2017_KatalonyaReferandumu",     "year": "2017-2018", "lang": "es", "mb_country": "ES", "limit": 200},
]

LANG_TO_FILE = {
    "tr": os.path.join(DATA_DIR, "data_tr.csv"),
    "en": os.path.join(DATA_DIR, "data_en.csv"),
    "ja": os.path.join(DATA_DIR, "data_ja.csv"),
    "fr": os.path.join(DATA_DIR, "data_fr.csv"),
    "es": os.path.join(DATA_DIR, "data_es.csv"),
}

MAX_SONGS_PER_ARTIST = 15
MIN_LYRICS_LENGTH    = 100
CHUNK_SIZE           = 50_000
MB_MAX_PER_YEAR      = 400   
MB_PER_ARTIST        = 100   
MAX_LRCLIB_CALLS     = 1000  

BASELINE_LOOKBACK_YEARS = 5
BASELINE_LIMIT          = 200

BLACKLIST_KEYWORDS = [
    "instrumental", "karaoke", "cover", "remix",
    "version", "live", "edit", "remaster",
]

LANG_TO_MB = {
    "tr": "tur", "en": "eng", "ja": "jpn", "fr": "fra", "es": "spa",
}

SEED_ARTISTS: dict[str, list[str]] = {
    "tr": [
        "Tarkan", "Sertab Erener", "Sezen Aksu", "Ezhel", "Sagopa Kajmer",
        "Patron", "Duman", "Ceza", "Mabel Matiz", "Gazapizm", "Şanışer",
        "Anıl Piyancı", "Tepki", "Norm Alaçam", "Semicenk", "Hadise",
        "Emre Aydın", "Haluk Levent", "Mor ve Ötesi", "Teoman",
        "Şebnem Ferah", "Sıla", "Kenan Doğulu", "Gülşen", "Ziynet Sali",
        "Murda", "Uzi", "Joker", "Ados", "Hidra", "Ben Fero", "Defkhan",
        "Massaka", "Contra", "Khontkar", "Cem Adrian", "Rafet El Roman",
        "İbrahim Tatlıses", "Hande Yener", "Yıldız Tilbe", "Nilüfer",
        "maNga", "Athena", "Pinhani", "Adamlar", "Yüzyüzeyken Konuşuruz",
        "Dolu Kadehi Ters Tut", "Göksel", "Funda Arar", "Feridun Düzağaç",
        "Edis", "Zeynep Bastık", "Killa Hakan", "Allame", "Şehinşah",
        "No.1", "Baneva", "Eypio", "Hayki", "Kayra", "Yalın", "Gökhan Türkmen",
        "Cem Karaca", "Barış Manço", "Müslüm Gürses", "Gökhan Özen"
    ],
    "en": [
        "Eminem", "Jay-Z", "Beyoncé", "Kanye West", "Alicia Keys",
        "50 Cent", "Nelly", "Usher", "Maroon 5", "Kelly Clarkson",
        "Taylor Swift", "Katy Perry", "Lady Gaga", "Rihanna", "Drake",
        "Kendrick Lamar", "Bruno Mars", "Ariana Grande", "Post Malone",
        "Billie Eilish", "Olivia Rodrigo", "The Weeknd", "Doja Cat",
        "Coldplay", "Arctic Monkeys", "Ed Sheeran", "Adele", "Dua Lipa",
        "Harry Styles", "Justin Bieber", "Snoop Dogg", "J. Cole",
        "Travis Scott", "Mac Miller", "Foo Fighters", "Red Hot Chili Peppers",
        "Linkin Park", "Green Day", "Imagine Dragons", "Lana Del Rey",
        "SZA", "Childish Gambino", "Tyler, The Creator"
    ],
    "ja": [
        "YOASOBI", "Official髭男dism", "King Gnu", "Ado", "Fujii Kaze",
        "Kenshi Yonezu", "Aimer", "LiSA", "Yorushika", "Kana Boon",
        "Mrs. GREEN APPLE", "back number", "Bump of Chicken", "ONE OK ROCK",
        "Perfume", "Radwimps", "Vaundy", "Creepy Nuts", "Eve", "Zutomayo",
        "あいみょん", "Hikaru Utada", "Namie Amuro", "Ayumi Hamasaki", 
        "Arashi", "AKB48", "L'Arc~en~Ciel", "X Japan", "BABYMETAL", 
        "B'z", "Mr.Children", "SPYAIR", "UVERworld", "Asian Kung-Fu Generation",
        "Macaroni Empitsu", "Saucy Dog", "KANA-BOON"
    ],
    "fr": [
        "Stromae", "Angèle", "Aya Nakamura", "Grand Corps Malade", "Louane",
        "Clara Luciani", "Soprano", "PNL", "Jul", "Ninho", "Nekfeu",
        "Orelsan", "Bigflo & Oli", "Christine and the Queens", "Pomme",
        "Vianney", "Lomepal", "Damso", "Gims", "M Pokora",
        "Booba", "Kaaris", "Niska", "Gazo", "Tiakola", "Zaz",
        "Kendji Girac", "Indochine", "Daft Punk", "Justice",
        "David Guetta", "DJ Snake", "Zazie", "Renaud", "Johnny Hallyday",
        "Mylène Farmer", "Amel Bent", "Vitaa", "SCH"
    ],
    "es": [
        "Rosalía", "Alejandro Sanz", "Pablo Alborán", "Rozalén", "Melendi",
        "Malú", "David Bisbal", "Ana Mena", "C. Tangana", "Bad Gyal",
        "Dani Martín", "Vetusta Morla", "Love of Lesbian", "Funambulista",
        "Enrique Iglesias", "Shakira", "Chenoa", "El Canto del Loco",
        "Estopa", "Amaral", "La Oreja de Van Gogh", "Fito & Fitipaldis",
        "Leiva", "Pereza", "Manolo García", "Vanesa Martín", "India Martínez",
        "Natos y Waor", "Ayax y Prok", "SFDK", "Kase.O", "Rels B", 
        "Quevedo", "Morad", "Lola Índigo", "Aitana", "Maka", "Beret", "Macaco"
    ]
}

def parse_year_range(s: str) -> list[str]:
    p = s.split("-")
    try:
        return [str(y) for y in range(int(p[0]), int(p[1]) + 1)]
    except Exception:
        return [s]

def is_blacklisted(title: str) -> bool:
    l = title.lower()
    pattern = r'\b(' + '|'.join(BLACKLIST_KEYWORDS) + r')\b'
    return bool(re.search(pattern, l))

def normalize(s: str) -> str:
    tr_map = {
        "I": "i", "ı": "i", "İ": "i",
        "Ğ": "g", "ğ": "g",
        "Ü": "u", "ü": "u",
        "Ş": "s", "ş": "s",
        "Ö": "o", "ö": "o",
        "Ç": "c", "ç": "c"
    }
    for k, v in tr_map.items():
        s = s.replace(k, v)
        
    nfkd = unicodedata.normalize("NFKD", s.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_title(title: str) -> str:
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"\[.*?\]", "", title)
    return title.split("-")[0].strip()

class GatekeeperAgent:
    def __init__(self, lang: str):
        self.lang = lang

    def evaluate(self, lyrics: str) -> tuple[bool, str]:
        if not lyrics or len(lyrics) < MIN_LYRICS_LENGTH:
            return False, f"kisa({len(lyrics) if lyrics else 0})"
        if self.lang == "ja":
            if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", lyrics):
                return True, "ja"
            return False, "ja-yok"
        try:
            det = detect(lyrics)
            return (True, det) if det == self.lang else (False, f"dil:{det}")
        except Exception:
            return False, "unknown"

# 1 — LOKAL DATASET
def load_local_rows(lang_file: str, years: list[str]) -> list[dict]:
    rows = []
    for chunk in pd.read_csv(lang_file, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.lower()
        a = next((c for c in ["artist","artist_name","singer"] if c in chunk.columns), None)
        t = next((c for c in ["title","song","song_name","track"] if c in chunk.columns), None)
        l = next((c for c in ["lyrics","lyric","text"] if c in chunk.columns), None)
        y = next((c for c in ["year","release_year","date"] if c in chunk.columns), None)
        if not (a and t and l and y):
            continue
        chunk = chunk.dropna(subset=[a, t, l, y])
        chunk["_y"] = chunk[y].astype(str).str.extract(r"(\d{4})")[0]
        chunk = chunk[chunk["_y"].isin(years)]
        df_sub = chunk[[a, t, l, "_y"]].astype(str)
        for row in df_sub.itertuples(index=False):
            rows.append({
                "artist": row[0].strip(),
                "title":  row[1].strip(),
                "lyrics": row[2].strip(),
                "year":   row[3],
            })
    return rows

# 2 — MB (rlang) + LOKAL INDEX
_local_index_cache: dict[str, dict[str, str]] = {}

def get_local_index(lang_file: str, lang: str) -> dict[str, str]:
    if lang in _local_index_cache:
        return _local_index_cache[lang]
    print(f"  [Indeks] {os.path.basename(lang_file)} okunuyor...")
    index: dict[str, str] = {}
    for chunk in pd.read_csv(lang_file, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.lower()
        a = next((c for c in ["artist","artist_name","singer"] if c in chunk.columns), None)
        t = next((c for c in ["title","song","song_name","track"] if c in chunk.columns), None)
        l = next((c for c in ["lyrics","lyric","text"] if c in chunk.columns), None)
        if not (a and t and l):
            continue
        chunk = chunk.dropna(subset=[a, t, l])
        df_sub = chunk[[a, t, l]].astype(str)
        for row in df_sub.itertuples(index=False):
            lyrics = row[2].strip()
            if len(lyrics) < MIN_LYRICS_LENGTH:
                continue
            artist = row[0].strip()
            title  = row[1].strip()
            for tk in [title, clean_title(title)]:
                key = f"{normalize(artist)}|||{normalize(tk)}"
                if key not in index:
                    index[key] = lyrics
    print(f"  [Indeks] {len(index):,} giris hazir.")
    _local_index_cache[lang] = index
    return index

def lookup_local(artist: str, title: str, index: dict) -> str | None:
    na = normalize(artist)
    for tk in [normalize(title), normalize(clean_title(title))]:
        v = index.get(f"{na}|||{tk}")
        if v:
            return v
    return None

_mb_rlang_cache: dict[str, list[dict]] = {}

def get_mb_rlang_candidates(years: list[str], mb_country: str, lang: str) -> list[dict]:
    mb_lang   = LANG_TO_MB.get(lang, "")
    cache_key = f"rl_{mb_lang}|{mb_country}|{'|'.join(sorted(years))}"
    if cache_key in _mb_rlang_cache:
        return _mb_rlang_cache[cache_key]

    candidates = []
    for year in years:
        if mb_lang and lang == "en":
            query = f"rlang:{mb_lang} AND country:{mb_country} AND date:{year}"
        elif mb_lang:
            query = f"rlang:{mb_lang} AND date:{year}"
        else:
            query = f"country:{mb_country} AND date:{year}"

        offset, fetched = 0, 0
        while fetched < MB_MAX_PER_YEAR:
            try:
                res  = musicbrainzngs.search_recordings(query=query, limit=100, offset=offset)
                recs = res.get("recording-list", [])
                if not recs:
                    break
                for rec in recs:
                    title = rec.get("title", "").strip()
                    if not title or is_blacklisted(title):
                        continue
                    for credit in rec.get("artist-credit", []):
                        if isinstance(credit, dict) and "artist" in credit:
                            candidates.append({"artist": credit["artist"]["name"],
                                               "title": title, "year": year})
                            break
                fetched += len(recs)
                total    = int(res.get("recording-count", 0))
                offset  += 100
                if offset >= min(total, MB_MAX_PER_YEAR):
                    break
                time.sleep(1.1)
            except Exception as e:
                print(f"  [MB-T2] Hata ({year}): {e}")
                time.sleep(3)
                break

    random.shuffle(candidates)
    _mb_rlang_cache[cache_key] = candidates
    return candidates

# 3 — SEED_ARTISTS × MB (artist+yil) → LRCLIB
_mb_artist_cache: dict[str, list[dict]] = {}

def get_mb_artist_candidates(years: list[str], lang: str) -> list[dict]:
    cache_key = f"art_{lang}|{'|'.join(sorted(years))}"
    if cache_key in _mb_artist_cache:
        print(f"  [MB-T3] Cache'den {len(_mb_artist_cache[cache_key])} aday.")
        return _mb_artist_cache[cache_key]

    seeds = SEED_ARTISTS.get(lang, [])
    if not seeds:
        print(f"  [MB-T3] '{lang}' icin seed listesi bos.")
        return []

    candidates = []
    print(f"  [MB-T3] {len(seeds)} sanatci x {len(years)} yil sorgulanıyor...")

    for artist_name in seeds:
        for year in years:
            query = f'artist:"{artist_name}" AND date:{year}'
            try:
                res  = musicbrainzngs.search_recordings(query=query, limit=MB_PER_ARTIST, offset=0)
                recs = res.get("recording-list", [])
                for rec in recs:
                    title = rec.get("title", "").strip()
                    if not title or is_blacklisted(title):
                        continue
                    for credit in rec.get("artist-credit", []):
                        if isinstance(credit, dict) and "artist" in credit:
                            candidates.append({
                                "artist": credit["artist"]["name"],
                                "title":  title,
                                "year":   year,
                            })
                            break
                time.sleep(1.1)
            except Exception as e:
                print(f"  [MB-T3] {artist_name}/{year} hata: {e}")
                time.sleep(3)

    random.shuffle(candidates)
    print(f"  [MB-T3] {len(candidates)} yil-dogru aday hazir.")
    _mb_artist_cache[cache_key] = candidates
    return candidates


def fetch_lyrics_lrclib(artist: str, title: str) -> str | None:
    try:
        resp = lrclib_session.get(
            "https://lrclib.net/api/get",
            params={"track_name": title, "artist_name": artist},
            timeout=10,
        )
        if resp.status_code == 200:
            data  = resp.json()
            plain = (data.get("plainLyrics") or "").strip()
            if plain:
                return plain
            synced = (data.get("syncedLyrics") or "").strip()
            if synced:
                lines = re.sub(r"\[\d+:\d+\.\d+\]\s*", "", synced).strip()
                return lines if lines else None
    except Exception:
        pass
    return None

def try_add(
    artist: str, title: str, lyrics: str, year: str,
    event_name: str, song_data: list, artist_counts: dict,
    existing_keys: set, limit: int, csv_path: str,
    agent: GatekeeperAgent, source: str,
    expected_country: str, lang: str, seed_set: set = None
) -> bool:
    if len(song_data) >= limit:
        return False
    if is_blacklisted(title):
        return False
    key = f"{artist.lower()}|||{title.lower()}"
    if key in existing_keys:
        return False
    if artist_counts.get(artist, 0) >= MAX_SONGS_PER_ARTIST:
        return False
    is_ok, _ = agent.evaluate(lyrics)
    if not is_ok:
        return False

    norm_name = artist.lower().strip()
    if seed_set is None:
        seed_set = {s.lower().strip() for s in SEED_ARTISTS.get(lang, [])}
    if norm_name not in seed_set:
        country = get_artist_country(artist)
        if not country or country.upper() != expected_country.upper():
            existing_keys.add(key) 
            return False

    song_data.append({"event": event_name, "year": year,
                       "artist": artist, "title": title, "lyrics": lyrics})
    existing_keys.add(key)
    artist_counts[artist] = artist_counts.get(artist, 0) + 1
    print(f"  [+] ({len(song_data):>3}/{limit}) [{source}] {artist} — {title}")
    
    if len(song_data) % 5 == 0:
        pd.DataFrame(song_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
        
    return True

def fetch_event(event: dict) -> None:
    event_name  = event["event_name"]
    year_str    = event["year"]
    lang        = event["lang"]
    mb_country  = event["mb_country"]
    limit       = event["limit"]
    years       = parse_year_range(year_str)
    lang_file   = LANG_TO_FILE.get(lang, "")
    csv_path    = os.path.join(RAW_DIR, f"{event_name}_raw.csv")

    print(f"\n{'=' * 62}")
    print(f"  {event_name}  [{year_str}]  ({lang.upper()})")
    print(f"{'=' * 62}")

    song_data: list[dict] = []
    artist_counts: dict[str, int] = {}
    existing_keys: set[str] = set()

    if os.path.exists(csv_path):
        try:
            df_ex = pd.read_csv(csv_path)
            song_data = df_ex.to_dict("records")
            for item in song_data:
                a, t = str(item.get("artist","")), str(item.get("title",""))
                artist_counts[a] = artist_counts.get(a, 0) + 1
                existing_keys.add(f"{a.lower()}|||{t.lower()}")
            if len(song_data) >= limit:
                print(f"  [TAMAMLANDI] {len(song_data)}/{limit}")
                return
            print(f"  [DEVAM] Mevcut: {len(song_data)}/{limit}")
        except Exception:
            pass

    agent = GatekeeperAgent(lang)
    seed_set = {s.lower().strip() for s in SEED_ARTISTS.get(lang, [])}

    # 1: LOKAL DATASET
    if lang_file and os.path.exists(lang_file):
        print(f"\n  [Tier 1] Lokal dataset...")
        local_rows = load_local_rows(lang_file, years)
        random.shuffle(local_rows)
        print(f"  Yil filtresi sonrasi: {len(local_rows)} satir")
        for row in local_rows:
            if len(song_data) >= limit:
                break
            try_add(row["artist"], row["title"], row["lyrics"], row["year"],
                    event_name, song_data, artist_counts, existing_keys,
                    limit, csv_path, agent, "T1:local", mb_country, lang, seed_set)

    # 2: MB (rlang) + LOKAL INDEX
    if len(song_data) < limit and lang_file and os.path.exists(lang_file):
        print(f"\n  [Tier 2] MB+lokal ({limit - len(song_data)} eksik)...")
        local_index   = get_local_index(lang_file, lang)
        mb_candidates = get_mb_rlang_candidates(years, mb_country, lang)
        for cand in mb_candidates:
            if len(song_data) >= limit:
                break
            lyrics = lookup_local(cand["artist"], cand["title"], local_index)
            if lyrics:
                try_add(cand["artist"], cand["title"], lyrics, cand["year"],
                        event_name, song_data, artist_counts, existing_keys,
                        limit, csv_path, agent, "T2:MB+local", mb_country, lang, seed_set)

    # 3: SEED_ARTISTS × MB + LRCLIB
    if len(song_data) < limit:
        need = limit - len(song_data)
        print(f"\n  [Tier 3] SEED_ARTISTS × MB + LRCLIB ({need} eksik)...")
        t3_candidates = get_mb_artist_candidates(years, lang)

        lrclib_calls = 0
        for cand in t3_candidates:
            if len(song_data) >= limit:
                break
            if lrclib_calls >= MAX_LRCLIB_CALLS:
                print(f"  [T3] {MAX_LRCLIB_CALLS} LRCLIB istek limitine ulasildi.")
                break
            key = f"{cand['artist'].lower()}|||{cand['title'].lower()}"
            if key in existing_keys:
                continue
            if artist_counts.get(cand["artist"], 0) >= MAX_SONGS_PER_ARTIST:
                continue
            if is_blacklisted(cand["title"]):
                continue

            lyrics = fetch_lyrics_lrclib(cand["artist"], cand["title"])
            lrclib_calls += 1

            if lyrics:
                ok = try_add(cand["artist"], cand["title"], lyrics, cand["year"],
                             event_name, song_data, artist_counts, existing_keys,
                             limit, csv_path, agent, "T3:LRCLIB", mb_country, lang, seed_set)
                if not ok:
                    existing_keys.add(key)

        print(f"  LRCLIB: {lrclib_calls} istek")

    print(f"\n  {'─' * 50}")
    if song_data:
        pd.DataFrame(song_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [KAYDEDILDI] {event_name} -> {len(song_data)} sarki")
    else:
        print(f"  {event_name} icin hic sarki toplanamadi!")


def generate_baseline_events(events: list[dict], lookback: int = BASELINE_LOOKBACK_YEARS) -> list[dict]:
    baseline_events = []
    for ev in events:
        year_range = ev["year"].split("-")
        try:
            start_year = int(year_range[0])
        except ValueError:
            continue

        baseline_end   = start_year - 1
        baseline_start = start_year - lookback

        if baseline_start < 1950:
            baseline_start = 1950

        baseline_year_str = f"{baseline_start}-{baseline_end}"

        baseline_event = {
            "event_name":  ev["event_name"] + "_BASELINE",
            "year":        baseline_year_str,
            "lang":        ev["lang"],
            "mb_country":  ev["mb_country"],
            "limit":       BASELINE_LIMIT,
        }
        baseline_events.append(baseline_event)
    return baseline_events


def fetch_data() -> None:
    load_artist_country_cache()

    print("\n" + "=" * 62)
    print("  Veri Toplanıyor")
    print("=" * 62)
    for event in TARGET_EVENTS:
        fetch_event(event)

    print("\n\n" + "=" * 62)
    print("  Baseline")
    print("=" * 62)
    baseline_events = generate_baseline_events(TARGET_EVENTS)
    for bl_event in baseline_events:
        fetch_event(bl_event)

    print("\n\nTum olaylar ve baseline verileri tamamlandi!")


if __name__ == "__main__":
    fetch_data()
