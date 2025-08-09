import os
import csv
import logging
import json
import string
import time
from collections import defaultdict

# Scientific and Data Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

import sqlite3
from datetime import datetime

from csv_writers import (write_word_pair_scores, write_word_averages)
from csv_readers import (read_word_averages)
from scoring import (candidate_gen, normalize_phoneme, _score_candidate)

# Constants and Settings
LETTERS = list(string.ascii_uppercase)
WORD_PAIR_FILENAME_TEMPLATE:str = "word_pairs_{0}_{1}_data.csv"
PHONEME_COORDINATE_DISTANCE_FILENAME:str = "phoneme_coordinate_distance.csv"
PHONEME_AUDIO_DISTANCE_FILENAME:str = "phoneme_audio_distance.csv"
LETTER_PAIR_FILENAME:str = "letter_pair_averages.csv"

# Phoneme Coordinates
PHONEME_DICT = defaultdict(list)
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
# TODO: Convert to JSON User Settings
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

# TODO: Convert to configurable user settings in the JSON
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

# TODO: Convert to configurable user settings in the JSON
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

# --------------
# Main Search Function
# --------------

""" Find the best set of words via random sampling. """
def find_best_set_randomized(words_by_letter, trials=1000, preselected_words=None, top_candidates=100, save_each_candidate=False):

    logging.info("Starting Randomized Search for Best Set of Words...")
    
    trial_name = f"random_search_{trials}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        results = _score_candidate(selected_words=c, phoneme_suffix_length=2, weights=None)
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
        
        if len(top_candidates) > TOP_N:
            top_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_candidates.pop()

        if save_each_candidate:
            save_candidates_to_db([candidate_entry], trial_name=trial_name)

    # Write all top candidates to CSV
    logging.info("Writing Top Candidates to CSV")
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



    logging.info(f"Top {TOP_N} candidates saved to 'best_random_search.csv'")

    # Save to database before returning

    save_candidates_to_db(top_candidates, trial_name=trial_name)

    # Return best scores in original format for compatibility
    best_scores = {
        "levenshtein": (top_candidates[0]["total_levenshtein"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "phoneme": (top_candidates[0]["total_phoneme_distance"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "seq": (top_candidates[0]["shared_sequence"], top_candidates[0]["candidate"]) if top_candidates else (float('inf'), []),
        "suffix": (top_candidates[0]["shared_suffix"], top_candidates[0]["candidate"]) if top_candidates else (float('inf'), []),
        "score": (top_candidates[0]["score"], top_candidates[0]["candidate"]) if top_candidates else (float('-inf'), []),
        "avg_calc_time": np.mean(calculation_times) if calculation_times else 0.0
    }

    return best_scores

def save_candidates_to_db(top_candidates, trial_name="random_search", db_path="phoneme_data.db"):
    """Save alphabet candidates to the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    batch_data = []
    
    for candidate in top_candidates:
        # Pad candidate list to 26 words if needed
        candidate_words = []
        for l in LETTERS:
            for w in candidate:
                if w.startswith(l):
                    candidate_words.append(w)
                    break
            else:
                candidate_words.append("")  # Fill with empty string if no word found

        batch_data.append((
            trial_name,
            candidate["score"],
            candidate["total_levenshtein"],
            candidate["total_phoneme_distance"],
            candidate["shared_sequence"],
            candidate["shared_suffix"],
            *candidate_words  # Unpack the 26 words
        ))

    cur.executemany("""
        INSERT INTO alphabet_candidates (
            trial_name, total_score,
            total_levenshtein, total_phoneme_distance, shared_sequences, shared_suffixes,
            word_a, word_b, word_c, word_d, word_e, word_f, word_g, word_h, word_i, word_j,
            word_k, word_l, word_m, word_n, word_o, word_p, word_q, word_r, word_s, word_t,
            word_u, word_v, word_w, word_x, word_y, word_z
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch_data)
    
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(batch_data)} candidates to database")

def get_candidates_from_db(metric_type=None, limit=10, db_path="phoneme_data.db"):
    """Retrieve candidates from the database."""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT * FROM alphabet_candidates 
        WHERE (? IS NULL OR metric_type = ?)
        ORDER BY
            rank_position
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=[metric_type, metric_type, limit])
    conn.close()
    
    return df

def get_alphabet_from_candidate(candidate_row):
    """Extract the 26-word alphabet from a candidate row."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    alphabet = {}
    
    for letter in letters:
        word = candidate_row[f'word_{letter}']
        if word:
            alphabet[letter.upper()] = word
    
    return alphabet

# -------------------------------
# Dictionary Functions
# -------------------------------

""" Clean the CMU Pronouncing Dictionary and apply filters.
- Filters out words with non-alphabetic characters, too short/long words, blacklisted words, etc.
- Lemmatizes words to their base forms and groups them by lemma, selecting the shortest variant.
- Adds custom words with predefined phonemes.
"""
def get_cleaned_cmu_dict():
    logging.info("Cleaning CMU Dictionary...")

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
                # TODO: Redo this filtering system because if any of the filters are false, 
                # then the word is not added instead of that filter being ignored, D'OH
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

    lemmatizer = WordNetLemmatizer()
    lemma_map = defaultdict(list)
    
    # Lemmatize the words, group by lemma (in cleaned dictionary)
    for w, prons in tqdm(cleaned_dict.items(), desc="Lemmatizing...", unit=" word"):
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
    
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detail, WARNING for less
        format='[%(levelname)s]: %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            # Uncomment below to also log to a file:
            # logging.FileHandler("phonetic_alphabet.log")
        ]
    )
    logging.info("Starting Best Phonetic Alphabet Utility")
    
    logging.info("Downloading CMU Dictionary")
    nltk.download('cmudict')
    nltk.download('wordnet')

    # TODO: Rewrite so that it utilizes the phoneme_data.db stuff
    PHONEME_DICT = get_cleaned_cmu_dict()
    PHONEME_DICT_NORMALIZED = {w: normalize_phoneme(PHONEME_DICT[w][0]) for w in PHONEME_DICT.keys()}

    WORDS_BY_LETTER = defaultdict(list)
    for word in tqdm(PHONEME_DICT.keys(), desc="Grouping Words by First Letter", unit="word"):
        WORDS_BY_LETTER[word[0].upper()].append(word)

    choices = {
        '1': "Generate Word Pair Scores",
        '2': "Generate Word Averages",
        '3': "Find Best (Randomized Trial)",
        '4': "Score Premade Alphabet"
    }

    logging.info("Best Phonetic Alphabet Utility")
    print("Choose an option:")
    for key, value in choices.items():
        print(f"{key}. {value}")
    
    user_input = input(f"Enter your choice ({choices.keys()}): ").strip().lower()
    
    choice = choices[user_input] if user_input in choices else None
    
    if not choice:
        print("Invalid choice. Please run the program again.")
        return

    if choice == "Generate Word Pair Scores":
        logging.info("Generating Word Pair Scores")
        print("\n---------------WARNING-----------------")
        print("\nThis operation will take a long time and will generate a LARGE number of BIG .csv files!!")
        print("\nPress Enter to continue or [Ctrl+C] to cancel.")
        input()
        print("---------------------------------")
        logging.info("Generating Word Pair Scores")
        write_word_pair_scores(p_dict=PHONEME_DICT, p_dict_norm=PHONEME_DICT_NORMALIZED, words_by_letters=WORDS_BY_LETTER, csv_headers=CSV_WORD_PAIR_HEADERS, filename_template=WORD_PAIR_FILENAME_TEMPLATE)

    elif choice == "Generate Word Averages":
        write_word_averages(p_dict=PHONEME_DICT, p_dict_norm=PHONEME_DICT_NORMALIZED, words_by_letter=WORDS_BY_LETTER, csv_headers=CSV_WORD_AVERAGE_HEADERS)

    elif choice == "Find Best (Randomized Trial)":
        logging.info("Finding Best Phonetic Alphabet via Randomized Trial...")

        TRIALS = 10000 #load_user_settings()["trials"]
        best_scores = find_best_set_randomized(words_by_letter=WORDS_BY_LETTER, trials=TRIALS, top_candidates=100)
        logging.info(f"Best Scores Found in {TRIALS} Trials")
        logging.info(f"Average Calculation Time: {best_scores['avg_calc_time']:.6f} seconds")

        logging.info("--------------------------------")
        logging.info(f"Best Overall Set ({best_scores['score']:,.6f}):")
        logging.info("--------------------------------")
        for word in best_scores['score'][1]:
            logging.info(f"{word.capitalize():-12s}  ->  {' '.join(PHONEME_DICT[word][0])}")

    elif choice == "Score Premade Alphabet":
        NATO = [w for w in NATO_PHONETIC_ALPHABET if w in PHONEME_DICT]
        best_scores = _score_candidate(selected_words=NATO, phoneme_suffix_length=2, weights=None)
        logging.info("--------------------------------")
        logging.info(f"NATO Score ({best_scores['score']:,.2f}):")
        logging.info("--------------------------------")
        for word in NATO:
            logging.info(f"{word.capitalize():-12s}  ->  {' '.join(PHONEME_DICT[word][0])}")

    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()