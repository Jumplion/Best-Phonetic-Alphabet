import sqlite3
import re
import urllib.request
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import tqdm 
import numpy as np
import typing
from g2p_en import G2p

# ============================================================
# Configuration
# ============================================================
DB_PATH = "BestPhonetics.db"

SQL_SCHEMA_PHONEMES = Path(__file__).resolve().parent / 'sql/schema_phonemes.sql'
SQL_SCHEMA_WORDS = Path(__file__).resolve().parent / 'sql/schema_words.sql'
SQL_SCHEMA_WORD_PAIRS = Path(__file__).resolve().parent / 'sql/schema_word_pairs.sql'

CMU_FILE = Path("word dumps/cmudict-0.7b")
CMU_URL = "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"

WIKTIONARY_FILE = Path("word dumps/enWikiDump.jsonl")
WIKTIONARY_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"

# Single lemmatizer instance (expensive to create repeatedly)
LEMMATIZER = WordNetLemmatizer()

# Batch size for bulk word insertion (tuneable)
BATCH_SIZE = 1000

# Precompiled regex for digit/stress stripping (micro-optimisation)
_DIGIT_RE = re.compile(r"\d")

PHONEMES: dict[str, tuple[float, float, float, float]] = {
            # VOWELS    
        # Vowel | Height | Backness| Roundness
        #   0 = Low [Open], 0.5 = Mid, 1 = High [Close]
        #   0 = Front,  0.5 = Central, 1 = Back
        #   0 = Rounded, 1 = Unrounded
        "AA":  (1,      0,    1,     0),    # ɑ             father
        "AE":  (1,      0,    0,     0),    # æ             cat
        "AH":  (1,      0.5,  0.5,   0),    # ʌ or ə        cut
        "AO":  (1,      0.5,  1,     1),    # ɔ`            caught
        "AW":  (1,      0.5,  0.75,  1),    # aʊ            cow
        "AX":  (1,      0.5,  0.5,   0),    # ə (schwa)     about
        "AY":  (1,      0.5,  0.5,   0),    # aɪ            my
        "EY":  (1,      0.65, 0,     0),    # e             they
        "EH":  (1,      0.5,  0,     0),    # ɛ             bed
        "ER":  (1,      0.5,  0.5,   0),    # ɚ or ɝ        her
        "IY":  (1,      1,    0,     0),    # i             see
        "IH":  (1,      0.85, 0,     0),    # ɪ             sit
        "OW":  (1,      0.65, 1,     1),    # o             go
        "OY":  (1,      0.5,  0.5,   0.5),  # ɔɪ            toy
        "UW":  (1,      1,    1,     1),    # u             too
        "UH":  (1,      0.85, 1,     1),    # ʊ             put

            # CONSONANTS
        # Consonant | Place of Articulation | Manner of Articulation | Voiced/Unvoiced
        # 0 = Bilabial, 0.2 = Labiodental, 0.4 = Dental, 0.6 = Alveolar, 0.8 = Velar, 1 = Glottal
        # 0 = Stop, 0.125 = Affricate, 0.25 = Fricative, 0.5 = Nasal, 0.75 = Lateral Liquid, 0.875 = Rhotic Liquid, 1 = Glide
        # 0 = Voiceless, 1 = Voiced
        
        # Stops
        "P":  (0, 0,    0,      0),  # voiceless bilabial stop              pat
        "B":  (0, 0,    0,      1),  # voiced bilabial stop                 bat
        "D":  (0, 0.4,  0,      1),  # voiced alveolar stop                 dog
        "T":  (0, 0.4,  0,      0),  # voiceless alveolar stop              top
        "K":  (0, 0.8,  0,      0),  # voiceless velar stop                 cat
        "G":  (0, 0.8,  0,      1),  # voiced velar stop                    go
        
        # Affricates
        "CH": (0, 0.6,  0.125,  0),  # voiceless postalveolar affricate     chip
        "JH": (0, 0.6,  0.125,  1),  # voiced postalveolar affricate        judge

        # Fricatives
        "F":  (0, 0.2,  0.25,   0),  # voiceless labiodental fricative      fish
        "V":  (0, 0.2,  0.25,   1),  # voiced labiodental fricative         van
        "TH": (0, 0.4,  0.25,   0),  # voiceless dental fricative           thin
        "DH": (0, 0.4,  0.25,   1),  # voiced dental fricative              then
        "S":  (0, 0.4,  0.25,   0),  # voiceless alveolar fricative         see
        "Z":  (0, 0.4,  0.25,   1),  # voiced alveolar fricative            zoo
        "SH": (0, 0.6,  0.25,   0),  # voiceless postalveolar fricative     she
        "ZH": (0, 0.6,  0.25,   1),  # voiced postalveolar fricative        measure
        "HH": (0, 1,    0.25,   0),  # voiceless glottal fricative          he

        # Nasals
        "M":  (0, 0,    0.5,    1),  # bilabial nasal                       me
        "N":  (0, 0.4,  0.5,    1),  # alveolar nasal                       no
        "NG": (0, 0.8,  0.5,    1),  # velar nasal                          sing

        # Liquids
        "L":  (0, 0.4,  0.75,   1),  # alveolar lateral liquid              leaf    
        "R":  (0, 0.4,  0.875,  1),  # alveolar rhotic liquid               red

        # Glides (approximants)
        "Y":  (0, 0.6,  1,      1),  # palatal glide    (IPA: /j/)          yes
        "W":  (0, 0,    1,      1)   # bilabial glide   (IPA: /w/)          we
    }

""" Orthographic letter prefixes that are banned. These are often silent letters at the start of words."""
BANNED_LETTER_PREFIXES: list[str] = [
    "eu",
    "gn",
    "kn",
    "mn",
    "ph",
    "pn",
    "ps",
    "sh",
    "th",
    "wr"
]

""" Phoneme Prefixes that are banned from certain letter groups.
For example, "E" words shouldn't start with a "y" sound like "eunuch" or "euphoria"
"""
BANNED_PHONEME_PREFIXES: dict[str, list[str]] = {
    "a" : [],
    "b" : [],
    "c" : ["S"],        # "cereal", "cell"
    "d" : [],
    "e" : ["ER", "Y"],  # "ernest" | "eunuch", "euphoria"
    "f" : [],
    "g" : ["N"],        # "gnome"
    "h" : ["AW", "OW"], # "hour"
    "i" : [],
    "j" : [],
    "k" : ["N"],        # "knight"
    "l" : [],
    "m" : ["N"],        # "mnemonic"
    "n" : [],
    "o" : [],
    "p" : ["F"],        # "phone", "philosophy"
    "q" : [],
    "r" : [],
    "s" : ["SH"],       # "she", "sure"
    "t" : ["TH", "DH"], # "this" | "that"
    "u" : [],
    "v" : [],    
    "w" : ["R", "HH"],  # 'wrestle' | 'whole'
    "x" : [],
    "y" : [],
    "z" : []
}

# ARPAbet vowel phonemes (with stress)
VOWELS: list[str] = [k for k, v in PHONEMES.items() if v[0] == 1]
CONSONANTS: list[str] = [k for k, v in PHONEMES.items() if v[0] == 0]
# For O(1) membership checks when counting syllables etc.
VOWEL_SET: set[str] = set(VOWELS)

ORTHOGRAPHIC_VOWELS: set[str] = set("AEIOUY")
ORTHOGRAPHIC_CONSONANTS: set[str] = set("BCDFGHJKLMNPQRSTVWXZ")

IPA_TO_ARPABET = {
    # Vowels
    "i": "IY", "iː": "IY", "ɪ": "IH", "e": "EH", "ɛ": "EH", "æ": "AE",
    "ɑ": "AA", "ɒ": "AA", "ɔ": "AO", "ʌ": "AH", "ə": "AH", "ɚ": "ER",
    "ɜː": "ER", "u": "UW", "uː": "UW", "ʊ": "UH", "oʊ": "OW", "o": "OW",
    "eɪ": "EY", "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY",
    # Consonants
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "r": "R", "ɹ": "R", "w": "W", "j": "Y",
    # Stress marks and punctuation (ignored)
    "ˈ": "", "ˌ": "", ".": "",
}

# Multi-character tokens should be matched first.
IPA_TOKENS = sorted(IPA_TO_ARPABET.keys(), key=lambda x: -len(x))

# Compile regex to match IPA symbols
IPA_PATTERN = re.compile("|".join(map(re.escape, IPA_TOKENS)))

g2p = G2p()

# ============================================================
# Download Functions
# ============================================================

def download_wordDump(url, outputFile):
    """Downloads a word dump from the given URL if not present locally."""
    if outputFile.exists():
        print(f"✅ {outputFile.name} already downloaded.")
        return
    print(f"⬇️ Downloading {outputFile.name}...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    outputFile.write_bytes(data)
    print("✅ Download complete.")


# ============================================================
# Phoneme and Word Helpers
# ============================================================

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


def ipa_to_arpabet(ipa_str: str) -> str:
    """Convert an IPA string to ARPAbet, preserving stress markers by attaching digits to vowel symbols.

    Primary stress (ˈ) -> 1, secondary (ˌ) -> 2. The digit is appended to the ARPAbet vowel token
    immediately following the stress marker.
    """
    if not ipa_str:
        return ""
    s = ipa_str.strip()
    # remove surrounding slash/brackets but keep inner stress marks
    s = re.sub(r'^[\[/]+|[\]/]+$', '', s)
    pos = 0
    arpabet_seq = []
    pending_stress = None
    # Tokenize using IPA_PATTERN
    while pos < len(s):
        match = IPA_PATTERN.match(s, pos)
        if match:
            token = match.group(0)
            pos = match.end()
            if token == 'ˈ':
                pending_stress = '1'
                continue
            if token == 'ˌ':
                pending_stress = '2'
                continue
            arp = IPA_TO_ARPABET.get(token, '')
            if not arp:
                # skip unknown
                continue
            # If this arp is a vowel (heuristic: vowels map to ARPAbet symbols containing letters AEIOU), attach stress digit
            if re.search(r'[AEIOU]', arp):
                if pending_stress:
                    arp = f"{arp}{pending_stress}"
                    pending_stress = None
                else:
                    # No lexical stress marker -> mark as unstressed with '0'
                    arp = f"{arp}0"
            arpabet_seq.append(arp)
        else:
            # skip a single character that didn't match
            pos += 1
    return ' '.join(arpabet_seq)


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


def parse_wiktionary_jsonl(line: str) -> typing.Optional[tuple[str, list[str]]]:
    """Parse a line of Wiktionary JSONL dump to extract word and IPA pronunciations."""
    import json
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    word = obj.get("word")
    ipa_list = obj.get("ipa", [])
    if not word or not ipa_list:
        return None

    arpabet_variants = []
    for ipa in ipa_list:
        arpabet = ipa_to_arpabet(ipa)
        if arpabet:
            arpabet_variants.append(arpabet)

    if not arpabet_variants:
        return None

    return word, arpabet_variants


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


def populate_phonemes(conn):
    phoneme_params = []
    for ph, features in tqdm.tqdm(PHONEMES.items(), desc="Inserting phonemes", unit="phoneme", mininterval=5, ncols=80, smoothing=0.1):
        phoneme_params.append(
            (
                ph,
                1 if ph in VOWEL_SET else 0,
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


def populate_phoneme_distances(conn):
    """Precompute and populate phoneme_distances table with pairwise distances."""
    phonemes = list(PHONEMES.keys())
    distance_params = []
    for i, ph1 in enumerate(tqdm.tqdm(phonemes, desc="Computing phoneme distances", unit="phoneme", mininterval=5, ncols=80, smoothing=0.1)):
        vec1 = np.array(PHONEMES[ph1])

        empty_vec = np.zeros(4)
        feature_dist_empty = np.linalg.norm(vec1 - empty_vec)
        audio_dist_empty = 0.0
        distance_params.append((ph1, "", feature_dist_empty, audio_dist_empty))

        for j in range(i, len(phonemes)):
            ph2 = phonemes[j]
            vec2 = np.array(PHONEMES[ph2])
            feature_dist = np.linalg.norm(vec1 - vec2)
            audio_dist = 0.0    # Placeholder for future audio-based distance

            distance_params.append((ph1, ph2, feature_dist, audio_dist))

    conn.executemany("""
        INSERT INTO phoneme_distances (phoneme_1, phoneme_2, feature_based_distance, audio_based_distance)
        VALUES (?, ?, ?, ?)
    """, distance_params)
    conn.commit()
    print(f"✅ Finished inserting {len(distance_params)} phoneme distances.")


def populate_words(conn):
    insert_sql = """
        INSERT INTO words (
            word, lemma, pronunciation_index,
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Calculate their stats and whatnot as well

    batch: list[tuple] = []
    wordSeen = {} # Count of how many times we've seen a word (for alternate pronunciations)
    total = 0

    # Parse and insert CMUdict data
    with CMU_FILE.open(encoding="latin-1") as f:
        for line in tqdm.tqdm(f, desc="Inserting CMU words...", unit=" word"):
            word, phonemes, pron_index = parse_cmudict_line(line)

            # Guard against static type checker warnings and malformed lines
            if not word or phonemes is None:
                continue

            if word in wordSeen.keys():
                continue
            else:
                wordSeen[word] = True

            pron_index = 0
            params = word_to_params(word, phonemes, pron_index, source='CMUdict-0.7b')
            if params != tuple():  # Skip invalid words
                batch.append(params)

            if len(batch) >= BATCH_SIZE:
                conn.executemany(insert_sql, batch)
                conn.commit()
                total += len(batch)
                batch.clear()

    print(f"Inserted {total} words from CMUdict.")

    # Parse and insert Wiktionary data
    with WIKTIONARY_FILE.open(encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Inserting Wiktionary words...", unit=" word"):
            
            parsed = parse_wiktionary_jsonl(line)
            if not parsed:
                continue
            word, arpabet_variants = parsed

            for pron_index, arpabet in enumerate(arpabet_variants):
                
                if word in wordSeen.keys():
                    continue
                else:
                    wordSeen[word] = True

                phonemes = arpabet.split()

                p_index = 0
                params = word_to_params(word, phonemes, p_index, source='Wiktionary')

                if params != tuple():  # Skip invalid words
                    batch.append(params)

            if len(batch) >= BATCH_SIZE:
                conn.executemany(insert_sql, batch)
                conn.commit()
                total += len(batch)
                batch.clear()

    print(f"Inserted {total} words from Wiktionary.")

    # Insert any remaining rows
    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)
        print(f"Inserted {total} words (final).")


def word_to_params(word: str, phonemes: list[str], pron_index: int, source: str = 'CMUdict-0.7b') -> tuple:
    """Return the parameter tuple for a single word insert (same order as INSERT)."""
    
    # Basic validation
    # If the word contains any non alphabetical characters. (Allow only A-Z and a-z)
    if not re.match(r"^[A-Za-z]+$", word):
        return tuple()
    # If the word is too short or too long...
    if len(word) < 3 or len(word) > 10:
        return tuple()
    # If the word consists of all the same letter...
    if all(c.lower() == word[0].lower() for c in word):
        return tuple()
    # If the word starts with a banned letter combination...
    if any(word.lower().startswith(prefix) for prefix in BANNED_LETTER_PREFIXES):
        return tuple()
    # If the word starts with a letter, and its phoneme list starts with a banned phoneme...
    if word[0].lower() in BANNED_PHONEME_PREFIXES:
        banned_prefixes = BANNED_PHONEME_PREFIXES[word[0].lower()]
        if phonemes and any(phonemes[0].startswith(bp) for bp in banned_prefixes):
            return tuple()
    # If the phoneme list is empty or less than 2 phonemes...
    if not phonemes or len(phonemes) < 2:
        return tuple()

    syllables = count_syllables(phonemes)
    # If the syllable count is outside 1-4 range...
    if syllables < 2 or syllables > 4:
        return tuple()
    
    possibleLemmas = []
    for pos in ['n', 'v', 'a', 'r']:
        possibleLemmas.append(LEMMATIZER.lemmatize(word.lower(), pos=pos))
        possibleLemmas.append(LEMMATIZER.lemmatize(possibleLemmas[-1], pos=pos))

    norm_phonemes = normalize_phoneme_list(phonemes)
    phoneme_count = len(phonemes)
    unique_phoneme_count = len(set(norm_phonemes))

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
        word.lower(), min(possibleLemmas, key=len), pron_index,
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
        word_frequency, phonotactic_probability, source
    )

# The population/main orchestration has been moved to `populate_database.py` to
# separate helper functions (download_cmudict, create_phoneme_db, create_word_db, etc.)
# from the script entry point. Import and call those helpers from other scripts.
def main():
    print("🛠️  Populating the BestPhonetics database...")
    
    # download_wordDump(CMU_URL, CMU_FILE)
    # download_wordDump(WIKTIONARY_URL, WIKTIONARY_FILE)
    
    conn = sqlite3.connect(DB_PATH)
    print("✅ Connected to database.")

    print("🗂️  Setting up database schema...")
    run_sql_file(conn, SQL_SCHEMA_PHONEMES)
    run_sql_file(conn, SQL_SCHEMA_WORDS)
    run_sql_file(conn, SQL_SCHEMA_WORD_PAIRS)

    print("Populating phonemes...")
    populate_phonemes(conn)

    print("Populating phoneme distances...")
    populate_phoneme_distances(conn)

    print("Populating words...")
    populate_words(conn)

    # After populating, one last cleanup step.
    # Group by lemma
    # Choose representative word for each lemma (shortest spelling, lowest pronunciation_index), remove the rest
    print("Cleaning up duplicates by lemma...")
    cleanup_sql = """
        DELETE FROM words
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM words
            GROUP BY lemma, pronunciation_index
        );
    """
    conn.execute(cleanup_sql)
    conn.commit()

    conn.close()

if __name__ == "__main__":
    main()