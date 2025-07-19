import os
import csv
import logging
import json
import string
import time
from collections import defaultdict

# Scientific and Data Libraries
import numpy as np
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

from csv_writers import (write_word_pair_scores, write_word_averages, write_phoneme_coord_distance_dict)
from csv_readers import (read_letter_pair_scores, read_word_averages, read_phoneme_difference_file)
from scoring import (build_phoneme_distance_matrix, candidate_gen, normalize_phoneme, _score_candidate)

# 🔧 CONFIGURATION
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detail, WARNING for less
    format='[%(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        # Uncomment below to also log to a file:
        # logging.FileHandler("phonetic_alphabet.log")
    ]
)

LETTERS = list(string.ascii_uppercase)
WORD_PAIR_FILENAME_TEMPLATE:str = "word_pairs_{0}_{1}_data.csv"
PHONEME_COORDINATE_DISTANCE_FILENAME:str = "phoneme_coordinate_distance.csv"
PHONEME_AUDIO_DISTANCE_FILENAME:str = "phoneme_audio_distance.csv"
LETTER_PAIR_FILENAME:str = "letter_pair_averages.csv"

# Phoneme Coordinates
PHONEME_DICT = defaultdict(list)
PHONEME_DICT_NORMALIZED = defaultdict(list)
PHONEME_DISTANCE_DICT = defaultdict()
PHONEME_AUDIO_DISTANCE_DICT = defaultdict()
PHONEME_COORD_MATRIX = None
PHONEME_AUDIO_MATRIX = None
PHONEME_MATRIX_INDEX = {}

WORDS_BY_LETTER = defaultdict(list)

# CSV Headers
CSV_WORD_PAIR_HEADERS:list[str] = [
    "Source", "Target", "Score",
    "Source-Phonemes", "Target-Phonemes",
    "Levenshtein Distance", "Phoneme Distance", 
    "Shared Sequence Count", "Sequences",
    "Suffix Count", "Suffixes",
    "Rhyme Count", "Rhymes",
    "Phoneme Audio Distance"
]
CSV_DISTANCE_AVERAGE_HEADERS:list[str] = [
    "Source", "Target", "Total Word Pairs",
    "Min Lev", "Max Lev", "Avg Lev", "Std Dev Lev",
    "Min Phoneme", "Max Phoneme", "Avg Phoneme", "Std Dev Phoneme",
    "Min Shared", "Max Shared", "Avg Shared", "Std Dev Shared",
    "Min Score", "Max Score", "Avg Score", "Std Dev Score"
]
CSV_WORD_AVERAGE_HEADERS:list[str] =  [
    "Word", "Letter", "Phonemes", "Phoneme_Count", "Phoneme_Magnitude", "Comparisons_Made",
    "Avg_Score", "Min_Score", "Max_Score", "Std_Score",
    "Avg_Levenshtein", "Min_Levenshtein", "Max_Levenshtein", "Std_Levenshtein",
    "Avg_Phoneme_Distance", "Min_Phoneme_Distance", "Max_Phoneme_Distance", "Std_Phoneme_Distance",
    "Avg_Shared_Sequences", "Min_Shared_Sequences", "Max_Shared_Sequences", "Std_Shared_Sequences",
    "Avg_Shared_Suffixes", "Min_Shared_Suffixes", "Max_Shared_Suffixes", "Std_Shared_Suffixes",
    "Avg_Rhymes", "Min_Rhymes", "Max_Rhymes", "Std_Rhymes"
]

# Word and Phoneme Blacklists and Whitelists
WORD_BLACKLIST = [
]
WORD_WHITELIST = [
]

# Custom words to add to the dictionary
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

NATO_PHONETIC_ALPHABET = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "x-ray", "yankee", "zulu"
]

""" Letter Prefixes that should not be at the start of words
These prefixes are often silent or not pronounced, so we filter them out
"""
BANNED_LETTER_PREFIXES = [
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
For example, "E" words shouldn't start with "y" like "eunuch" or "euphoria"
"""
BANNED_PHONEME_PREFIXES = {
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

# Phoneme Coordinates
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

# --------------
# Main Search Function
# --------------

""" Find the best set of words via random sampling. """
def find_best_set_randomized(words_by_letter, p_norm_dict, 
                                p_coordinates, p_indices, p_coord_matrix, p_audio_matrix,
                                trials=1000, preselected_words=None, top_candidates=100):

    log_console_header("Starting Randomized Search for Best Set of Words", trials)
    
    # Track top N candidates for each metric
    TOP_N = max(10, top_candidates)
    top_candidates = []

    # Handle preselected words
    preselected_by_letter = {}
    if preselected_words:
        for w in preselected_words:
            le = w[0].upper()
            preselected_by_letter[le] = min(preselected_by_letter[le], w) if le in preselected_by_letter else w

    calculation_times = []
    for c in tqdm(candidate_gen(trials, words_by_letter, preselected_by_letter), total=trials, desc="Generating and Scoring Candidates", unit=" candidate", colour="green"):
        start_time = time.time()
        results = _score_candidate(selected_words=c, p_norm_dict=p_norm_dict, 
                                      p_coordinates=p_coordinates, p_indices=p_indices,
                                      p_coord_matrix=p_coord_matrix, p_audio_matrix=p_audio_matrix,
                                      phoneme_suffix_length=2, weights=None)
        calculation_times.append(time.time() - start_time)
        
        candidate_entry = {
            "candidate": sorted(list(c)),
            "score": results["score"],
            "total_levenshtein": results["total_levenshtein"],
            "total_phoneme_distance": results["total_phoneme_distance"],
            "shared_sequence": results["shared_sequence"][0],
            "shared_suffix": results["shared_suffix"][0]
        }
        
        top_candidates.append(candidate_entry)
        top_candidates.sort(key=lambda x: x["score"], reverse=True)
        if len(top_candidates) > TOP_N:
            top_candidates.pop()

    # Write all top candidates to CSV
    log_console_header("Writing Top Candidates to CSV")
    headers = ["Score", "Total Levenshtein Distance", "Total Phoneme Distance", 
               "Total Shared Sequences", "Total Shared Suffixes"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    with open("best_random_search.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write top candidates for each metric
        for entry in top_candidates:
            row = [
                entry["score"],
                entry["total_levenshtein"],
                entry["total_phoneme_distance"],
                    entry["shared_sequence"],
                    entry["shared_suffix"]
                ] + entry["candidate"]
            writer.writerow(row)

    # Return best scores in original format for compatibility
    best_scores = {
        "levenshtein": (top_candidates[0]["total_levenshtein"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "phoneme": (top_candidates[0]["total_phoneme_distance"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "seq": (top_candidates[0]["shared_sequence"], top_candidates[0]["candidate"]) if top_candidates else (float('inf'), []),
        "suffix": (top_candidates[0]["shared_suffix"], top_candidates[0]["candidate"]) if top_candidates else (float('inf'), []),
        "score": (top_candidates[0]["score"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "avg_calc_time": np.mean(calculation_times) if calculation_times else 0.0
    }

    log_console_header(f"Top {TOP_N} candidates saved to 'best_random_search.csv'")
    return best_scores

# -------------------------------
# Dictionary Functions
# -------------------------------

def get_phoneme_coord_distance_dict():
    distances = read_phoneme_difference_file(PHONEME_COORDINATE_DISTANCE_FILENAME)
    if distances is None:
        logging.info("Phoneme Coordinate Distance Dictionary does not exist. Creating it now.")
        distances = write_phoneme_coord_distance_dict()
    return distances

def get_phoneme_audio_difference_dict():
    distances = read_phoneme_difference_file(PHONEME_AUDIO_DISTANCE_FILENAME)
    if distances is None:
        logging.info("Phoneme Audio Distance Dictionary does not exist. Run the 'connear_notebook.ipynb' to create it!")
        distances = {}
    return distances

""" Clean the CMU Pronouncing Dictionary and apply filters.
- Filters out words with non-alphabetic characters, too short/long words, blacklisted words, etc.
- Lemmatizes words to their base forms and groups them by lemma, selecting the shortest variant.
- Adds custom words with predefined phonemes.
"""
def get_cleaned_cmu_dict():

    # Get user settings
    settings = load_user_settings()
    
    # Get Settings
    min_phoneme_length = settings["min_phoneme_length"]
    max_phoneme_length = settings["max_phoneme_length"]
    min_word_length = settings["min_word_length"]
    max_word_length = settings["max_word_length"]
    min_syllables = settings["min_syllables"]
    max_syllables = settings["max_syllables"]

    # Get Filters
    filter_phoneme_length = settings["filter_phoneme_length"]
    filter_word_length = settings["filter_word_length"]
    filter_syllables = settings["filter_syllables"]
    filter_letter_prefixes = settings["filter_letter_prefixes"]
    filter_phoneme_prefixes = settings["filter_phoneme_prefixes"]
    filter_averages = settings["filter_averages"]

    word_averages = read_word_averages()
    if word_averages is None:
        logging.error("Word averages file not found. Dictionary won't filter out words based on averages.")
    min_avg = np.mean(list(word_averages.values())) if word_averages else 0.0

    cleaned_dict = cmudict.dict()

    cleaned_dict = {
        w: prons for w, prons in tqdm(cleaned_dict.items(), desc="Filtering Words from CMU Dictionary", unit=" word")
        if (
            (w in WORD_WHITELIST) or
            (
                # TODO: Redo this filtering system because if any of the filters are false, then the word is not added
                # instead of that filter being ignored, D'OH
                (w not in WORD_BLACKLIST)
                and (w[0].isalpha())
                and (wordnet.synsets(w))
                and (filter_phoneme_length      and (min_phoneme_length <= len(prons[0]) <= max_phoneme_length))
                and (filter_word_length         and (min_word_length <= len(w) <= max_word_length))
                and (filter_syllables           and (min_syllables <= len([p for p in prons[0] if p[-1].isdigit()]) <= max_syllables))
                and (filter_letter_prefixes     and not any(w.startswith(prefix) for prefix in BANNED_LETTER_PREFIXES))
                and (filter_phoneme_prefixes    and not any(w.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(w[0].lower(), [])))
                and (filter_averages            and (word_averages[w] >= min_avg if word_averages else True))
            )
        )
    }              
    logging.info(f"CMU Dictionary Filtered | Total Words: {len(cleaned_dict)}")

    lemmatizer = WordNetLemmatizer()
    lemma_map = defaultdict(list)
    
    # Lemmatize the words, group by lemma (in cleaned dictionary)
    for w, prons in tqdm(cleaned_dict.items(), desc="Lemmatizing......", unit=" word"):
        noun = lemmatizer.lemmatize(w.lower(), pos='n')
        lemma = lemmatizer.lemmatize(noun, pos='v')
        lemma_map[lemma].append((w, prons[0]))

    # Choose representative word for each lemma (shortest spelling)
    lemma_dict = {}
    for lemma, variants in lemma_map.items():
        representative = min(variants, key=lambda x: len(x[0]))
        lemma_dict[representative[0]] = [representative[1]]  # keep as list for CMU compatibility 
    
    cleaned_dict = lemma_dict

    # Add Custom Words to the dictionary
    for w, pron in CUSTOM_WORDS.items():
        cleaned_dict[w.lower()] = [pron]

    logging.info(f"CMU Pronouncing Dictionary Cleaned | Total Words: {len(cleaned_dict)}")
    return cleaned_dict

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def log_console_header(message, data=None):
    logging.info("--------------------------------")
    if data:
        logging.info("%s: %s", message, data)
    else:
        logging.info(message)
    logging.info("--------------------------------")

def log_scores(list_name, set, p_dict):
    logging.info("--------------------------------")
    logging.info("Best %s Set (%.6f):", list_name, set[0])
    logging.info("--------------------------------")
    for word in set[1]:
        logging.info("%-12s  ->  %s", word.capitalize(), ' '.join(p_dict[word][0]))

def load_user_settings(settings_path="user_settings.json"):
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    with open(settings_path, "r") as f:
        settings = json.load(f)
    return settings

# -------------------------------
# Main Function
# -------------------------------

def main():
    log_console_header("Loading and Cleaning CMU Dictionary")
    # Load the CMU Pronouncing Dictionary and create the phoneme distance dictionary
    nltk.download('cmudict')
    nltk.download('wordnet')

    PHONEME_DICT = get_cleaned_cmu_dict()
    PHONEME_DICT_NORMALIZED = {w: normalize_phoneme(PHONEME_DICT[w][0]) for w in PHONEME_DICT.keys()}

    logging.info("Total Words: {:,}".format(len(PHONEME_DICT)))
    
    PHONEME_DISTANCE_DICT = get_phoneme_coord_distance_dict()
    PHONEME_AUDIO_DISTANCE_DICT = get_phoneme_audio_difference_dict()
    PHONEME_COORD_MATRIX, PHONEME_MATRIX_INDEX = build_phoneme_distance_matrix(PHONEME_DISTANCE_DICT)
    PHONEME_AUDIO_MATRIX, _ = build_phoneme_distance_matrix(PHONEME_AUDIO_DISTANCE_DICT)

    logging.info("Phoneme Coordinate Distance Matrix: %s", PHONEME_COORD_MATRIX.shape)
    logging.info("Phoneme Audio Distance Matrix: %s", PHONEME_AUDIO_MATRIX.shape)

    WORD_PHONEME_INDICES = {w: np.array([PHONEME_MATRIX_INDEX.get(ph, -1) for ph in p1]) for w, p1 in PHONEME_DICT_NORMALIZED.items()}

    WORDS_BY_LETTER = defaultdict(list)
    for word in tqdm(PHONEME_DICT.keys(), desc="Grouping Words by First Letter", unit="word"):
        WORDS_BY_LETTER[word[0].upper()].append(word)

    choices = {
        '1': "Generate Word Pair Scores",
        '2': "Generate Word Averages",
        '3': "Find Best (Randomized Trial)",
        '4': "Score Premade Alphabet"
    }

    log_console_header("Best Phonetic Alphabet Utility")
    print("Choose an option:")
    for key, value in choices.items():
        print(f"{key}. {value}")
    
    user_input = input(f"Enter your choice ({choices.keys()}): ").strip().lower()
    
    choice = choices[user_input] if user_input in choices else None
    
    if not choice:
        print("Invalid choice. Please run the program again.")
        return

    if choice == "Generate Word Pair Scores":
        log_console_header("Generating Word Pair Scores")
        print("\n---------------WARNING-----------------")
        print("\nThis operation will take a long time and will generate a LARGE number of BIG .csv files!!")
        print("\nPress Enter to continue or [Ctrl+C] to cancel.")
        input()
        print("---------------------------------")
        log_console_header("Generating Word Pair Scores")
        write_word_pair_scores(p_dict=PHONEME_DICT, p_dict_norm=PHONEME_DICT_NORMALIZED, p_indices=WORD_PHONEME_INDICES,
                                p_coordinates=PHONEME_COORDINATES, p_coord_matrix=PHONEME_COORD_MATRIX, p_audio_matrix=PHONEME_AUDIO_MATRIX, 
                                words_by_letters=WORDS_BY_LETTER, csv_headers=CSV_WORD_PAIR_HEADERS, filename_template=WORD_PAIR_FILENAME_TEMPLATE)

    elif choice == "Generate Word Averages":
        write_word_averages(p_dict=PHONEME_DICT, p_dict_norm=PHONEME_DICT_NORMALIZED, p_coordinates=PHONEME_COORDINATES,
                            p_indices=WORD_PHONEME_INDICES, p_coord_matrix=PHONEME_COORD_MATRIX, p_audio_matrix=PHONEME_AUDIO_MATRIX,
                            words_by_letter=WORDS_BY_LETTER, csv_headers=CSV_WORD_AVERAGE_HEADERS)

    elif choice == "Find Best (Randomized Trial)":
        log_console_header("Finding Best Phonetic Alphabet via Randomized Trial...")

        TRIALS = 1000000 # load_user_settings()["trials"]
        best_scores = find_best_set_randomized(words_by_letter=WORDS_BY_LETTER, 
                                                p_norm_dict=PHONEME_DICT_NORMALIZED,
                                                p_coordinates=PHONEME_COORDINATES,
                                                p_coord_matrix=PHONEME_COORD_MATRIX,
                                                p_audio_matrix=PHONEME_AUDIO_MATRIX,
                                                p_indices=WORD_PHONEME_INDICES,
                                                trials=TRIALS, top_candidates=100)
        log_console_header(f"Best Scores Found in {TRIALS} Trials")
        logging.info("Average Calculation Time: %.6f seconds", best_scores['avg_calc_time'])
        
        # Log best sets
        log_scores("Overall", best_scores['score'], PHONEME_DICT)
    
    elif choice == "Score Premade Alphabet":
        NATO = [w for w in NATO_PHONETIC_ALPHABET if w in PHONEME_DICT]
        best_scores = _score_candidate(selected_words=NATO,
                                        p_norm_dict=PHONEME_DICT_NORMALIZED,
                                        p_coordinates=PHONEME_COORDINATES,
                                        p_coord_matrix=PHONEME_COORD_MATRIX,
                                        p_audio_matrix=PHONEME_AUDIO_MATRIX,
                                        p_indices=WORD_PHONEME_INDICES,
                                        phoneme_suffix_length=2, weights=None)
        logging.info("--------------------------------")
        logging.info(f"NATO Score ({best_scores['score']:,.2f}):")
        logging.info("--------------------------------")
        for word in NATO:
            logging.info("%-8s  ->  %s", word.capitalize(), ' '.join(PHONEME_DICT[word][0]))

    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()