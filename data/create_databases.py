import sqlite3
import re
import urllib.request
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import tqdm 
import numpy as np
import typing

# ============================================================
# Configuration
# ============================================================
DB_PATH = "BestPhonetics.db"
CMU_URL = "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
CMU_FILE = Path("cmudict-0.7b")

# Batch size for bulk word insertion (tuneable)
BATCH_SIZE = 1000

PHONEMES: dict[str, tuple[float, float, float, float]] = {
            # VOWELS    
        # Vowel | Height | Backness| Roundness
        #   0 = Low [Open], 0.5 = Mid, 1 = High [Close]
        #   0 = Front,  0.5 = Central, 1 = Back
        #   0 = Rounded, 1 = Unrounded
        "AA":  (0,      0,    1,     0),    # ɑ             father
        "AE":  (0,      0,    0,     0),    # æ             cat
        "AH":  (0,      0.5,  0.5,   0),    # ʌ or ə        cut
        "AO":  (0,      0.5,  1,     1),    # ɔ`            caught
        "AW":  (0,      0.5,  0.75,  1),    # aʊ            cow
        "AX":  (0,      0.5,  0.5,   0),    # ə (schwa)     about
        "AY":  (0,      0.5,  0.5,   0),    # aɪ            my
        "EY":  (0,      0.65, 0,     0),    # e             they
        "EH":  (0,      0.5,  0,     0),    # ɛ             bed
        "ER":  (0,      0.5,  0.5,   0),    # ɚ or ɝ        her
        "IY":  (0,      1,    0,     0),    # i             see
        "IH":  (0,      0.85, 0,     0),    # ɪ             sit
        "OW":  (0,      0.65, 1,     1),    # o             go
        "OY":  (0,      0.5,  0.5,   0.5),  # ɔɪ            toy
        "UW":  (0,      1,    1,     1),    # u             too
        "UH":  (0,      0.85, 1,     1),    # ʊ             put

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

# ============================================================
# Utility Functions
# ============================================================

# ARPAbet vowel phonemes (with stress)
VOWELS: list[str] = [k for k, v in PHONEMES.items() if v[0] == 0]
CONSONANTS: list[str] = [k for k, v in PHONEMES.items() if v[0] == 1]
# For O(1) membership checks when counting syllables etc.
VOWEL_SET: set[str] = set(VOWELS)

ORTHOGRAPHIC_VOWELS: set[str] = set("AEIOUY")
ORTHOGRAPHIC_CONSONANTS: set[str] = set("BCDFGHJKLMNPQRSTVWXZ")

def download_cmudict():
    """Downloads the CMU dictionary if not present locally."""
    if CMU_FILE.exists():
        print("✅ CMU Dictionary already downloaded.")
        return
    print("⬇️ Downloading CMU Dictionary...")
    with urllib.request.urlopen(CMU_URL) as resp:
        data = resp.read()
    CMU_FILE.write_bytes(data)
    print("✅ Download complete.")

# Single lemmatizer instance (expensive to create repeatedly)
LEMMATIZER = WordNetLemmatizer()

# Precompiled regex for digit/stress stripping (micro-optimisation)
_DIGIT_RE = re.compile(r"\d")

def is_vowel_phoneme(ph):
    """Check if a phoneme (e.g. AH0) is a vowel."""
    base = _DIGIT_RE.sub("", ph)
    return base in VOWEL_SET

def normalize_phoneme_list(phonemes):
    """Remove stress markers (numbers) from phonemes."""
    return [_DIGIT_RE.sub("", ph) for ph in phonemes]

def count_syllables(phonemes):
    """Count syllables = count of vowel phonemes."""
    return sum(1 for ph in phonemes if is_vowel_phoneme(ph))

def orth_vowel_consonant_counts(word):
    """Count vowels/consonants in spelling."""
    word = re.sub(r"[^A-Za-z]", "", word.upper())
    v = sum(1 for c in word if c in ORTHOGRAPHIC_VOWELS)
    c = sum(1 for c in word if c not in ORTHOGRAPHIC_VOWELS)
    return v, c

def phon_vowel_consonant_counts(phonemes):
    """Count vowels/consonants in pronunciation."""
    v = sum(1 for ph in phonemes if is_vowel_phoneme(ph))
    c = len(phonemes) - v
    return v, c

# ============================================================
# Database Setup
# ============================================================

def run_sql_file(conn, sql_path: Path):
    """Execute a .sql schema file on the given sqlite3 connection."""
    if not sql_path.exists():
        print(f"ℹ️ SQL file not found: {sql_path}")
        return False
    sql_text = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql_text)
    conn.commit()
    print(f"✅ Executed SQL schema from {sql_path}")
    return True

# ============================================================
# CMU Parsing & Insertion
# ============================================================

def parse_cmudict_line(line):
    """Parse a CMU line like: WORD(1)  W ER1 D"""
    if not line or line.startswith(";;;"):
        return None, None, None

    parts = line.strip().split("  ")
    if len(parts) != 2:
        return None, None, None

    word_part, phoneme_part = parts
    phonemes = phoneme_part.split()

    # handle alternate pronunciations like WORD(1)
    match = re.match(r"^([A-Z'._-]+)(\((\d+)\))?$", word_part)
    if not match:
        return None, None, None

    word = match.group(1)
    pron_index = int(match.group(3)) if match.group(3) else 0

    return word, phonemes, pron_index

def word_to_params(word: str, phonemes: list[str], pron_index: int) -> tuple:
    """Return the parameter tuple for a single word insert (same order as INSERT)."""
    noun = LEMMATIZER.lemmatize(word.lower(), pos='n')
    lemma = LEMMATIZER.lemmatize(noun, pos='v')

    norm_phonemes = normalize_phoneme_list(phonemes)
    phoneme_count = len(phonemes)
    unique_phoneme_count = len(set(norm_phonemes))
    syllables = count_syllables(phonemes)

    orth_v, orth_c = orth_vowel_consonant_counts(word)
    phon_v, phon_c = phon_vowel_consonant_counts(phonemes)

    orth_ratio = orth_v / orth_c if orth_c > 0 else None
    phon_ratio = phon_v / phon_c if phon_c > 0 else None
    g2p_ratio = len(word) / phoneme_count if phoneme_count > 0 else None

    stress_pattern = "".join("1" if '1' in ph else "0" if is_vowel_phoneme(ph) else "" for ph in phonemes)
    stress_indices = [i for i, ph in enumerate(phonemes) if '1' in ph]
    primary_stress_index = stress_indices[0] if stress_indices else None

    word_vowels = [ph for ph in norm_phonemes if is_vowel_phoneme(ph)]
    word_consonants = [ph for ph in norm_phonemes if not is_vowel_phoneme(ph)]

    avg_vowel_height = np.mean([PHONEMES[ph][1] for ph in word_vowels]) if word_vowels else None
    avg_vowel_backness = np.mean([PHONEMES[ph][2] for ph in word_vowels]) if word_vowels else None
    avg_vowel_roundness = np.mean([PHONEMES[ph][3] for ph in word_vowels]) if word_vowels else None
    avg_consonant_voicing = np.mean([PHONEMES[ph][1] for ph in word_consonants]) if word_consonants else None
    avg_consonant_place = np.mean([PHONEMES[ph][2] for ph in word_consonants]) if word_consonants else None
    avg_consonant_manner = np.mean([PHONEMES[ph][3] for ph in word_consonants]) if word_consonants else None

    sonority_profile = None
    sonority_rise_count = None
    sonority_fall_count = None
    word_frequency = None
    phonotactic_probability = None

    return (
        word, lemma, pron_index,
        " ".join(phonemes), " ".join(norm_phonemes),
        phoneme_count, unique_phoneme_count,
        syllables, primary_stress_index, stress_pattern,
        len(word),
        orth_v, orth_c,
        phon_v, phon_c,
        orth_ratio, phon_ratio, g2p_ratio,
        avg_vowel_height, avg_vowel_backness, avg_vowel_roundness,
        avg_consonant_voicing, avg_consonant_place, avg_consonant_manner,
        sonority_profile, sonority_rise_count, sonority_fall_count,
        word_frequency, phonotactic_probability, 'CMUdict-0.7b'
    )

# ============================================================
# Main Execution
# ============================================================

def main():
    print("🛠️  Creating and populating the BestPhonetics database...")
    download_cmudict()
    sql_file = Path("data/BestPhonetics.sql")
    conn = sqlite3.connect(DB_PATH)
    
    print("Attempting to run SQL File and create databases")
    run_sql_file(conn, sql_file)    # Create tables just incase

    # Populate the Phonemes table (batch insert)
    phoneme_params = []
    for ph, features in tqdm.tqdm(PHONEMES.items(), desc="Inserting phonemes", unit="phoneme", mininterval=5, ncols=80, smoothing=0.1):
        phoneme_params.append(
            (
                ph,
                1 if features[0] else 0,
                features[1] if ph in VOWEL_SET else None,
                features[2] if ph in VOWEL_SET else None,
                features[3] if ph in VOWEL_SET else None,
                features[1] if ph not in VOWEL_SET else None,
                features[2] if ph not in VOWEL_SET else None,
                features[3] if ph not in VOWEL_SET else None,
            )
        )
    conn.executemany("""
        INSERT INTO phonemes (
            phoneme,
            is_vowel,
            height, backness, roundness,
            place, manner, voicing
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, phoneme_params)
    
    conn.commit()
    print(f"✅ Finished inserting {len(PHONEMES)} phonemes.")

    # Populate the Words table from CMUdict
    # Calculate their stats and whatnot as well
    with CMU_FILE.open(encoding="latin-1") as f:
        for line in tqdm.tqdm(f, desc="Inserting words", unit="word", mininterval=5, ncols=80, smoothing=0.1):
            word, phonemes, pron_index = parse_cmudict_line(line)
            if not word:
                continue
        insert_sql = """
            INSERT INTO words (
                word, lemma, cmu_pron_index,
                phoneme_list, normalized_phoneme_list,
                phoneme_count, unique_phoneme_count, 
                num_syllables, primary_stress_index, stress_pattern,
                word_length, 
                orth_vowel_count, orth_consonant_count,
                phon_vowel_count, phon_consonant_count,
                orth_vowel_consonant_ratio, phon_vowel_consonant_ratio, grapheme_to_phoneme_ratio,
                avg_vowel_height, avg_vowel_backness, avg_vowel_roundness,
                avg_consonant_voicing, avg_consonant_place, avg_consonant_manner,
                sonority_profile, sonority_rise_count, sonority_fall_count,
                word_frequency, phonotactic_probability, source
            ) VALUES (%s)
        """
        # Build a proper placeholder string for sqlite (question marks)
        placeholder_count = 30  # number of columns in the INSERT above
        insert_sql = insert_sql.replace("(%s)", "(" + ", ".join(["?"] * placeholder_count) + ")")

        batch: list[tuple] = []
        total = 0
        with CMU_FILE.open(encoding="latin-1") as f:
            for line in tqdm.tqdm(f, desc="Inserting words", unit="word", mininterval=5, ncols=80, smoothing=0.1):
                word, phonemes, pron_index = parse_cmudict_line(line)
                if not word:
                    continue
                # Guard against static type checker warnings and malformed lines
                if phonemes is None:
                    continue
                if pron_index is None:
                    pron_index = 0
                params = word_to_params(word, phonemes, pron_index)
                batch.append(params)

                if len(batch) >= BATCH_SIZE:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    total += len(batch)
                    print(f"Inserted {total} words...")
                    batch.clear()

        # Insert any remaining rows
        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            total += len(batch)
            print(f"Inserted {total} words (final).")

        conn.close()

if __name__ == "__main__":
    main()