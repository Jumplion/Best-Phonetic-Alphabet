import sqlite3
import pandas as pd
import ast
import re
from scoring import normalize_phoneme
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

PHONEME_COORDINATES = {   
    # VOWELS    
    # Vowel |  Backness | Height | Roundness
    #   0 = Front,  0.5 = Central, 1 = Back
    #   0 = Low [Open], 0.5 = Mid, 1 = High [Close]
    #   0 = Rounded, 1 = Unrounded
    "AA":  (0,  1,      0,       0),    # ɑ             father
    "AE":  (0,  0,      0,       0),    # æ             cat
    "AH":  (0,  0.5,    0.5,     0),    # ʌ or ə        cut
    "AO":  (0,  1,      0.5,     1),    # ɔ`            caught
    "AW":  (0,  0.75,   0.5,     1),    # aʊ            cow
    "AX":  (0,  0.5,    0.5,     0),    # ə (schwa)     about
    "AY":  (0,  0.5,    0.5,     0),    # aɪ            my
    "EY":  (0,  0,      0.65,    0),    # e             they
    "EH":  (0,  0,      0.5,     0),    # ɛ             bed
    "ER":  (0,  0.5,    0.5,     0),    # ɚ or ɝ        her
    "IY":  (0,  0,      1,       0),    # i             see
    "IH":  (0,  0,      0.85,    0),    # ɪ             sit
    "OW":  (0,  1,      0.65,    1),    # o             go
    "OY":  (0,  0.5,    0.5,     0.5),  # ɔɪ            toy
    "UW":  (0,  1,      1,       1),    # u             too
    "UH":  (0,  1,      0.85,    1),    # ʊ             put
    
    # CONSONANTS
    # Consonant | Place of Articulation | Manner of Articulation | Voiced/Unvoiced
    # 0 = Bilabial, 0.2 = Labiodental, 0.4 = Dental, 0.6 = Alveolar, 0.8 = Velar, 1 = Glottal
    # 0 = Stop, 0.125 = Affricate, 0.25 = Fricative, 0.5 = Nasal, 0.75 = Lateral Liquid, 0.875 = Rhotic Liquid, 1 = Glide
    # 0 = Voiceless, 1 = Voiced
    
    # Stops
    "P":  (1, 0,    0,      0),  # voiceless bilabial stop              pat
    "B":  (1, 0,    0,      1),  # voiced bilabial stop                 bat
    "D":  (1, 0.4,  0,      1),  # voiced alveolar stop                 dog
    "T":  (1, 0.4,  0,      0),  # voiceless alveolar stop              top
    "K":  (1, 0.8,  0,      0),  # voiceless velar stop                 cat
    "G":  (1, 0.8,  0,      1),  # voiced velar stop                    go
    
    # Affricates
    "CH": (1, 0.6,  0.125,  0),  # voiceless postalveolar affricate     chip
    "JH": (1, 0.6,  0.125,  1),  # voiced postalveolar affricate        judge

    # Fricatives
    "F":  (1, 0.2,  0.25,   0),  # voiceless labiodental fricative      fish
    "V":  (1, 0.2,  0.25,   1),  # voiced labiodental fricative         van
    "TH": (1, 0.4,  0.25,   0),  # voiceless dental fricative           thin
    "DH": (1, 0.4,  0.25,   1),  # voiced dental fricative              then
    "S":  (1, 0.4,  0.25,   0),  # voiceless alveolar fricative         see
    "Z":  (1, 0.4,  0.25,   1),  # voiced alveolar fricative            zoo
    "SH": (1, 0.6,  0.25,   0),  # voiceless postalveolar fricative     she
    "ZH": (1, 0.6,  0.25,   1),  # voiced postalveolar fricative        measure
    "HH": (1, 1,    0.25,   0),  # voiceless glottal fricative          he

    # Nasals
    "M":  (1, 0,    0.5,    1),  # bilabial nasal                       me
    "N":  (1, 0.4,  0.5,    1),  # alveolar nasal                       no
    "NG": (1, 0.8,  0.5,    1),  # velar nasal                          sing

    # Liquids
    "L":  (1, 0.4,  0.75,   1),  # alveolar lateral liquid              leaf    
    "R":  (1, 0.4,  0.875,  1),  # alveolar rhotic liquid               red

    # Glides (approximants)
    "Y":  (1, 0.6,  1,      1),  # palatal glide    (IPA: /j/)          yes
    "W":  (1, 0,    1,      1)   # bilabial glide   (IPA: /w/)          we
}   

CUSTOM_WORDS = {
    "amogus": ["AH", "M", "OW", "G", "Y", "UW", "S"],
    "atrioc": ["AH", "T", "R", "IY", "AA", "K"],
    "based": ["B", "EY", "S", "T"],
    "bigchungus": ["B", "IH", "G", "CH", "AH", "NG", "G", "Y", "UW", "S"],
    "boomer": ["B", "UW", "M", "ER"],
    "boomers": ["B", "UW", "M", "ER", "Z"],
    "bruh": ["B", "R", "AH"],
    "chad": ["CH", "AE", "D"],
    "chads": ["CH", "AE", "D", "Z"],
    "cheems": ["CH", "IY", "M", "Z"],
    "choom": ["CH", "UW", "M"],
    "chungus": ["CH", "AH", "NG", "G", "Y", "UW", "S"],
    "cringe": ["K", "R", "IH", "N", "JH"],
    "cringed": ["K", "R", "IH", "N", "JH", "T"],
    "cringes": ["K", "R", "IH", "N", "JH", "IH", "Z"],
    "cringing": ["K", "R", "IH", "N", "JH", "IH", "NG"],
    "degen": ["D", "EH", "JH", "EH", "N"],
    "degens": ["D", "EH", "JH", "EH", "N", "Z"],
    "doge": ["D", "OW", "JH"],
    "fomo": ["F", "OW", "M", "OW"],
    "fud": ["F", "AH", "D"],
    "fudds": ["F", "AH", "D", "Z"],
    "glizzy": ["G", "L", "IH", "Z", "IY"],
    "gyatt": ["JH", "AY", "AE", "T"],
    "kappa": ["K", "AE", "P", "AH"],
    "kekw": ["K", "EH", "K", "D", "AH", "B", "L", "Y", "UW"],
    "lol": ["EH", "L", "OW", "EH", "L"],
    "lmao": ["EH", "L", "M", "EY", "OW"],
    "lurk": ["L", "ER", "K"],
    "noob": ["N", "UW", "B"],
    "noobs": ["N", "UW", "B", "Z"],
    "nocap": ["N", "OW", "K", "AE", "P"],
    "omegalul": ["OW", "M", "EH", "G", "AH", "L", "UW", "L"],
    "pepe": ["P", "EH", "P", "EY"],
    "pepega": ["P", "EH", "P", "EY", "G", "AH"],
    "pepegas": ["P", "EH", "P", "EY", "G", "AH", "Z"],
    "pepehands": ["P", "EH", "P", "EY", "HH", "AE", "N", "D", "Z"],
    "pog": ["P", "AO", "G"],
    "pogchamp": ["P", "AO", "G", "CH", "AE", "M", "P"],
    "pogchamps": ["P", "AO", "G", "CH", "AE", "M", "P", "S"],
    "poggers": ["P", "AO", "G", "ER", "Z"],
    "poggies": ["P", "AO", "G", "IY", "Z"],
    "pogu": ["P", "AO", "G", "Y", "UW"],
    "rekt": ["R", "EH", "K", "T"],
    "rofl": ["R", "OW", "F", "AH", "L"],
    "roflmao": ["R", "OW", "F", "AH", "L", "M", "EY", "OW"],
    "sheesh": ["SH", "IY", "SH"],
    "shrek": ["SH", "R", "EH", "K"],
    "shreking": ["SH", "R", "EH", "K", "IH", "NG"],
    "shreks": ["SH", "R", "EH", "K", "S"],
    "simp": ["S", "IH", "M", "P"],
    "sus": ["S", "AH", "S"],
    "tendie": ["T", "EH", "N", "D", "IY"],
    "tendies": ["T", "EH", "N", "D", "IY", "Z"],
    "ussy": ["AH", "S", "IY"],
    "xd": ["EH", "K", "S", "D", "IY"],
    "yeet": ["Y", "IY", "T"],
    "yikes": ["Y", "AY", "K", "S"],
    "yiker": ["Y", "AY", "K", "ER"],
    "zoinks": ["Z", "OY", "NG", "K", "S"],
    "zomg": ["Z", "OW", "M", "G"],
    "skibidi": ["S", "K", "IH", "B", "IY", "D", "IY"],
    "rizz": ["R", "IH", "Z"],
    "rizzed": ["R", "IH", "Z", "D"],
    "rizzler": ["R", "IH", "Z", "L", "ER"],
    "incel": ["IH", "N", "S", "EH", "L"],
    "obamna": ["OW", "B", "AA", "M", "N", "AH"],
    "glarketing": ["G", "L", "AA", "R", "K", "AH", "T", "IH", "NG"],
    "glarketer": ["G", "L", "AA", "R", "K", "AH", "T", "ER"],
    "alpha": ["AE", "L", "F", "AH"],
    "bravo": ["B", "R", "AE", "V", "OW"],
    "charlie": ["CH", "AA", "R", "L", "IY"],
    "delta": ["D", "EH", "L", "T", "AH"],
    "echo": ["EH", "K", "OW"],
    "foxtrot": ["F", "AA", "K", "S", "T", "R", "AA", "T"],
    "golf": ["G", "AO", "L", "F"],
    "hotel": ["HH", "OW", "T", "EH", "L"],
    "india": ["IH", "N", "D", "IY", "AH"],
    "iowa": ["AY", "OW", "AH"],
    "juliett": ["JH", "UW", "L", "IY", "EH", "T"],
    "kilo": ["K", "IY", "L", "OW"],
    "lima": ["L", "IY", "M", "AH"],
    "mike": ["M", "AY", "K"],
    "michelle": ["M", "IH", "SH", "EH", "L"],
    "micheal": ["M", "AY", "K", "AH", "L"],
    "november": ["N", "OW", "V", "EH", "M", "B", "ER"],
    "oscar": ["AA", "S", "K", "ER"],
    "papa": ["P", "AH", "P", "AH"],
    "quebec": ["K", "W", "EH", "B", "EH", "K"],
    "romeo": ["R", "OW", "M", "IY", "OW"],
    "romero": ["R", "OW", "M", "EH", "R", "OW"],
    "sierra": ["S", "IH", "EH", "R", "AH"],
    "tango": ["T", "AE", "NG", "G", "OW"],
    "uniform": ["Y", "UW", "N", "AH", "F", "AO", "R", "M"],
    "victor": ["V", "IH", "K", "T", "ER"],
    "whiskey": ["W", "IH", "S", "K", "IY"],
    "xray": ["EH", "K", "S", "R", "EY"],
    "yankee": ["Y", "AE", "NG", "K", "IY"],
    "zulu": ["Z", "UW", "L", "UW"],
    "zangeif": ["Z", "AE", "N", "G", "EY", "F"]
}

# --- File paths ---
output_db_path = "phoneme_data.db"

nltk.download('cmudict')
nltk.download('wordnet')

# --- Connect to SQLite database ---
conn = sqlite3.connect(output_db_path)
cur = conn.cursor()

# --- Create tables ---
cur.execute("""
CREATE TABLE IF NOT EXISTS cmudict (
    word TEXT PRIMARY KEY,
    pronunciation TEXT,
    norm_pron TEXT,
    syllables INTEGER,
    lemma TEXT,
    avg_score REAL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS alphabet_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trial_name TEXT,
    total_score REAL,
    total_levenshtein REAL,
    total_phoneme_distance REAL,
    shared_sequences INTEGER,
    shared_suffixes INTEGER,
    word_a TEXT, word_b TEXT, word_c TEXT, word_d TEXT, word_e TEXT,
    word_f TEXT, word_g TEXT, word_h TEXT, word_i TEXT, word_j TEXT,
    word_k TEXT, word_l TEXT, word_m TEXT, word_n TEXT, word_o TEXT,
    word_p TEXT, word_q TEXT, word_r TEXT, word_s TEXT, word_t TEXT,
    word_u TEXT, word_v TEXT, word_w TEXT, word_x TEXT, word_y TEXT, word_z TEXT
);
""")

# --- Optimize database settings ---
cur.execute("PRAGMA journal_mode = WAL")
cur.execute("PRAGMA synchronous = NORMAL") 
cur.execute("PRAGMA cache_size = 1000000")
cur.execute("PRAGMA temp_store = MEMORY")

# --- Load CMU dictionary ---
cmu_dict = cmudict.dict()
lemmatizer = WordNetLemmatizer()

# --- Populate cmudict with batching ---
batch_data = []
batch_size = 1000

cur.execute("BEGIN TRANSACTION")

for w in tqdm(cmu_dict.keys(), desc="Populating cmudict table"):
    pronunciation_string = " ".join(cmu_dict[w][0])
    norm_pron = normalize_phoneme(cmu_dict[w][0])
    norm_pron_string = " ".join(norm_pron) if norm_pron else ""
    syllables = len([p for p in cmu_dict[w][0] if p[-1].isdigit()]) if cmu_dict[w][0] else 0

    noun = lemmatizer.lemmatize(w.lower(), pos='n')
    lemma = lemmatizer.lemmatize(noun, pos='v')

    batch_data.append((
        w.lower(),
        pronunciation_string,
        norm_pron_string,
        syllables,
        lemma,
        0.0
    ))
    
    if len(batch_data) >= batch_size:
        cur.executemany("""
            INSERT OR REPLACE INTO cmudict (word, pronunciation, norm_pron, syllables, lemma, avg_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch_data)
        batch_data = []

# Insert remaining data
if batch_data:
    cur.executemany("""
        INSERT OR REPLACE INTO cmudict (word, pronunciation, norm_pron, syllables, lemma, avg_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, batch_data)

cur.execute("COMMIT")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_candidates_score ON alphabet_candidates(total_score);
""")

# --- Finalize ---
conn.commit()
conn.close()

print(f"✅ Database created at '{output_db_path}'")
