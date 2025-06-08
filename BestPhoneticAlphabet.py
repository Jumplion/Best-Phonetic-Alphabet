import os
import csv
import math
import random
import logging
import itertools
import json
import string
import time
import igraph as ig
import matplotlib, matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures
import glob

import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

import editdistance

import plotly.graph_objects as go
import networkx as nx

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

FILENAME_CSV_TEMPLATE = "word_pairs_{0}_{1}_data.csv"

# Words that we don't want in our alphabets
# These are words that are either offensive, inappropriate, or just don't fit the criteria
# Some other reasons for blacklisting words:
# - Not caught by the filters
# - Strange pronunciations that don't fit the phonetic alphabet
# - Words that are too similar to other words in the alphabet
# - Slurs and racial remarks are a no-go, we diverse and tolerant in this sum-bitch
WORD_BLACKLIST = [
]

WORD_WHITELIST = [
]

# Custom words to add to the dictionary
# These words will be added to the dictionary with their specified phonemes
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
    "glizzytober": ["G", "L", "IH", "Z", "IY", "T", "OW", "B", "ER"],
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
    "poguette": ["P", "AO", "G", "Y", "UW", "EH", "T"],
    "poguettes": ["P", "AO", "G", "Y", "UW", "EH", "T", "S"],
    "pogus": ["P", "AO", "G", "Y", "UW", "Z"],
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
}

# Prefixes that should not be at the start of words
# These prefixes are often silent or not pronounced, so we filter them out
PREFIX_FILTERS = [
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

# Prefix filters for phonemes
# Certain phonemes should not be at the start of words in certain letter groups
# For example, "E" words shouldn't start with "y" like "eunuch" or "euphoria"
BANNED_PHONEME_PREFIXES = {
    "a" : [],
    "b" : [],
    "c" : ["S"],
    "d" : [],
    "e" : ["ER", "Y"],
    "f" : [],
    "g" : [],
    "h" : [],
    "i" : [],
    "j" : [],
    "k" : [],
    "l" : [],
    "m" : [],
    "n" : [],
    "o" : [],
    "p" : ["F"],
    "q" : [],
    "r" : [],
    "s" : ["SH"],
    "t" : ["TH", "DH"],
    "u" : [],
    "v" : [],    
    "w" : ["R"],
    "x" : [],
    "y" : [],
    "z" : []
}

# Phoneme Coordinates
PHONEME_COORDINATES = {   
    # VOWELS    
    # Vowel |  Backness (Front 0 / Central 0.5 / Back 1)     Height (Low [Open] 0 / Mid 0.5 / High [Close] 1)      Roundness (Rounded 0 / Unrounded 1)
    "AA":  (0,  1,      0,       0),    # ɑ             father
    "AE":  (0,  0,      0,       0),    # æ             cat
    "AH":  (0,  0.5,    0.5,     0),    # ʌ or ə        cut
    "AO":  (0,  1,      0.5,     1),    # ɔ`            caught
    "AW":  (0,  0.75,   0.5,     1),    # aʊ            cow
    "AX":  (0,  0.5,    0.5,     0),    # ə (unstressed)
    "AY":  (0,  0.5,    0.5,     0),    # aɪ            my
    "EY":  (0,  0,      0.65,    0),    # e             they
    "EH":  (0,  0,      0.5,     0),    # ɛ             bed
    "ER":  (0,  0.5,    0.5,     0),    # ɚ or ɝ        her
    "IY":  (0,  0,      1,       0),    # i             see
    "IH":  (0,  0,      0.85,    0),    # ɪ             sit
    "OW":  (0,  1,      0.65,    1),    # o             go
    "OY":  (0,  0.5,    0.5,     0.5),  # ɔɪ        
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
PHONEME_DICT = defaultdict(list)
PHONEME_DISTANCE_DICT = {}
WORDS_BY_LETTER = defaultdict(list)
SHOW_DICTIONARY_STATS = False       # Set to 'True' to show stats about the dictionary after cleanup

# -----------------------------
# Phoneme and Word Length Options
# -----------------------------
MIN_PHONEME_LENGTH = 2      # Minimum number of phonemes required per word
MAX_PHONEME_LENGTH = 10     # Maximum number of phonemes allowed per word
MIN_WORD_LENGTH = 3         # Minimum number of letters required per word
MAX_WORD_LENGTH = 10        # Maximum number of letters allowed per word
MIN_SYLLABLES = 2           # Minimum number of syllables allowed per word
MAX_SYLLABLES = 3           # Maximum number of syllables allowed per word

# -----------------------------
# 📦 FUNCTION DEFINITIONS
# -----------------------------

# --------------------------------
# Calculation Functions
# --------------------------------

# Calculate Levenshtein distance between two words.
def levenshtein_distance(w1, w2):
    return editdistance.eval(w1, w2)

# Calculate the phoneme distance between two phoneme sequences.
def phoneme_distance(p1_list, p2_list, p_distance_dict):
    distance = 0
    for i in range(min(len(p1_list), len(p2_list))):
        distance += p_distance_dict.get((p1_list[i], p2_list[i]), 0)

    # Handle the case where one list is longer than the other
    remaining_phonemes = p1_list[len(p2_list):] if len(p1_list) > len(p2_list) else p2_list[len(p1_list):]
    # If the remaining phonemes are empty, return the distance
    if not remaining_phonemes:
        return distance
    # If the remaining phonemes are not empty, add the distance of the remaining phonemes
    # Adding the distance/magnitude of remaining phonemes are a flawed system, need to figure out a better way to handle this   
    distance += sum(p_distance_dict.get((remaining_phonemes[i], ""), 0) for i in range(len(remaining_phonemes)))   
    return distance

# Calculate the number of shared phoneme sequences between two phoneme sequences.
# Count the number of shared contiguous phoneme subarrays between p1 and p2,
# for all lengths from 1 up to N, where N = min(len(p1), len(p2)).
# Takes p_distance_dict purely for compatibility, but does not use it in this function.
def shared_phoneme_sequences(p1, p2, p_distance_dict): 
    set1, set2 = set(), set()
    n1, n2 = len(p1), len(p2)
    N = min(n1, n2)

    # Generate all contiguous subarrays of length 1 to N for p1
    for length in range(1, N + 1):
        for i in range(n1 - length + 1):
            set1.add(tuple(p1[i:i + length]))

    # Generate all contiguous subarrays of length 1 to N for p2
    for length in range(1, N + 1):
        for i in range(n2 - length + 1):
            set2.add(tuple(p2[i:i + length]))

    # Count the intersection
    return len(set1 & set2)

def _score_candidate_unpack(args):
    return _score_candidate(*args)

# Helper function for scoring a candidate.
def _score_candidate(candidate, p_dict, p_distance_dict):
    pairs = list(itertools.combinations(candidate, 2))
    lev = phon = shared = []

    for w1, w2 in pairs:      
        p1, p2 = p_dict[w1][0], p_dict[w2][0]
        
        lev.append(levenshtein_distance(w1, w2))
        phon.append(phoneme_distance(p1, p2, p_distance_dict))
        shared.append(shared_phoneme_sequences(p1, p2, p_distance_dict))
    
    avg_levenshtein = np.mean(lev)
    avg_phoneme = np.mean(phon)
    avg_shared = np.mean(shared)
    score = (avg_levenshtein * LEV_WEIGHT) + (avg_phoneme * PHONEME_WEIGHT) - (avg_shared * SHARED_WEIGHT)
    del pairs, lev, phon, shared
    return (score, avg_levenshtein, avg_phoneme, avg_shared)

# Find the best set of words via random sampling (no parallel processing).
def find_best_set_randomized(p_dict, p_distance_dict, words_by_letter, trials=1000, batch_size=1000): 
    log_console_header("Starting Randomized Search for Best Set of Words", trials)
    best_scores = {
        "score": (float('-inf'), []),
        "levenshtein": (float('-inf'), []),
        "phoneme": (float('-inf'), []),
        "shared": (float('inf'), []),
    }

    letters = list(string.ascii_uppercase)

    # Delete the file if it exists
    if os.path.exists("random_search_log.csv"):
        os.remove("random_search_log.csv")

    def candidate_gen():
        for _ in range(trials):
            c = [random.choice(words_by_letter[l]) for l in letters if words_by_letter[l]]
            if len(c) == len(letters):  # Ensure we have one word per letter
                yield c

    def update_bests(score, avg_lev, avg_phon, avg_shared, c):
        best_scores["levenshtein"] = max(best_scores["levenshtein"], (avg_lev, c), key=lambda x: x[0])
        best_scores["phoneme"] = max(best_scores["phoneme"], (avg_phon, c), key=lambda x: x[0])
        best_scores["shared"] = min(best_scores["shared"], (avg_shared, c), key=lambda x: x[0])
        best_scores["score"] = max(best_scores["score"], (score, c), key=lambda x: x[0])
    
    with open("random_search_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Score", "Avg Levenshtein Distance", "Avg Phoneme Distance (Basic)", "Avg Shared Phoneme Sequence Count"
        ] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

        batch = []
        for c in tqdm(candidate_gen(), total=trials, desc="Generating and Scoring Candidates", unit="candidate"):
            batch.append(c)
            if len(batch) >= batch_size:
                # Build a minimal p_dict for this batch
                all_words = set(w for candidate in batch for w in candidate)
                mini_p_dict = {w: p_dict[w] for w in all_words}
                # Prepare arguments for each candidate
                args = [(candidate, mini_p_dict, p_distance_dict) for candidate in batch]
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = list(tqdm(executor.map(_score_candidate_unpack, args), total=len(args), desc="Scoring Candidates in Batch", unit="candidate", leave=False))
                for c, (score, avg_lev, avg_phon, avg_shared) in zip(batch, results):
                    writer.writerow([score, avg_lev, avg_phon, avg_shared] + list(c))
                    update_bests(score, avg_lev, avg_phon, avg_shared, c)
                batch = []
        # Process any remaining
        if batch:
            all_words = set(w for candidate in batch for w in candidate)
            mini_p_dict = {w: p_dict[w] for w in all_words}
            args = [(candidate, mini_p_dict, p_distance_dict) for candidate in batch]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(_score_candidate_unpack, args), total=len(args), desc="Scoring Candidates in Batch", unit="candidate", leave=False))
            for c, (score, avg_lev, avg_phon, avg_shared) in zip(batch, results):
                writer.writerow([score, avg_lev, avg_phon, avg_shared] + list(c))
                update_bests(score, avg_lev, avg_phon, avg_shared, c)

    return best_scores

# Given a list of unique, pre-selected words, fill out the rest of the alphabet with random sampling,
# and return the best phonetic alphabet found after the specified number of trials.
def find_best_set_with_preselected(
    preselected_words,
    p_dict,
    p_distance_dict,
    words_by_letter,
    trials=1000,
    batch_size=1000
):

    # Normalize and deduplicate preselected words
    preselected_words = [w.lower() for w in preselected_words]
    preselected_by_letter = {}
    for w in preselected_words:
        l = w[0].upper()
        if l in preselected_by_letter:
            print(f"Warning: Multiple preselected words for letter '{l}'. Using '{min(preselected_by_letter[l], w)}'.")
            preselected_by_letter[l] = min(preselected_by_letter[l], w)
        else:
            preselected_by_letter[l] = w

    letters = list(string.ascii_uppercase)
    # Check that all preselected words exist in the dictionary
    for l, w in preselected_by_letter.items():
        if w not in p_dict:
            raise ValueError(f"Preselected word '{w}' for letter '{l}' not found in dictionary.")

    # Prepare the pool of available words for each letter (excluding preselected)
    available_words_by_letter = {
        l: [w for w in words_by_letter[l] if w not in preselected_by_letter.values()]
        for l in letters
    }

    # Build the fixed part of the candidate
    fixed_words = [preselected_by_letter[l] for l in letters if l in preselected_by_letter]
    fixed_letters = set(preselected_by_letter.keys())

    best_candidate = None
    best_details = None

    def candidate_gen():
        for _ in range(trials):
            candidate = fixed_words.copy()
            used_letters = fixed_letters.copy()
            for l in letters:
                if l not in used_letters:
                    candidate.append(random.choice(available_words_by_letter[l]))
            if len(candidate) == len(letters):
                yield candidate

    def process_candidate(b):
        # Build a minimal p_dict for this batch
        all_words = set(w for candidate in b for w in candidate)
        mini_p_dict = {w: p_dict[w] for w in all_words}
        args = [(candidate, mini_p_dict, p_distance_dict) for candidate in b]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_score_candidate_unpack, args), total=len(args), desc="Scoring Candidates in Batch", unit="candidate", leave=False))
        for candidate, (score, avg_lev, avg_phon, avg_shared) in zip(b, results):
            if score > best_details[0] if best_details else float('-inf'):
                best_candidate = candidate
                best_details = (score, avg_lev, avg_phon, avg_shared)    
        return best_candidate, best_details
    
    batch = []
    for c in tqdm(candidate_gen(), total=trials, desc="Generating and Scoring Candidates", unit="candidate", colour="green"):
        batch.append(c)
        if len(batch) >= batch_size:
            best_candidate, best_details = process_candidate(batch)
            batch = []
    # Process any remaining
    if batch:
        best_candidate, best_details = process_candidate(batch)

    # Print the best result
    print("\nBest Phonetic Alphabet Found:")
    print(f"Score: {best_details[0]:.4f} | Avg Levenshtein: {best_details[1]:.4f} | Avg Phoneme: {best_details[2]:.4f} | Avg Shared: {best_details[3]:.4f}")
    for l, w in zip(letters, best_candidate):
        tag = "(preselected)" if l in preselected_by_letter and preselected_by_letter[l] == w else ""
        print(f"{l}: {w} {tag}")

    return best_candidate, best_details

# Helper function for parallel processing of word distances.
def _word1_distances(args):
    word1, word1_pron, filtered_words, pron_dict, phoneme_distances = args
    total_dist_levenshtein = total_dist_phoneme = total_shared = count = 0.0
    
    for word2 in filtered_words:
        word2_pron = pron_dict[word2][0]
        total_dist_levenshtein += levenshtein_distance(word1, word2)
        total_dist_phoneme += phoneme_distance(word1_pron, word2_pron, phoneme_distances)
        total_shared += shared_phoneme_sequences(word1_pron, word2_pron, phoneme_distances)
        count += 1
    
    # Calculate the magnitude of the phoneme sequence
    magnitude = math.sqrt(sum(phoneme_distances.get((p, ""), 0) ** 2 for p in word1_pron))
    return magnitude, total_dist_levenshtein, total_dist_phoneme, total_shared, count

# ---------------------------------
# Calculate and write word averages to a CSV file.
# This function calculates the average Levenshtein distance, phoneme distance,
# and shared phoneme sequence count for each word, and writes the results to a CSV file.
# ---------------------------------
def write_word_averages(p_dict, p_distance_dict):

    # Filter Word lists excluding words starting with each letter
    word_list = list(p_dict.keys())
    filter_word_list = {
        "A": [w for w in word_list if w[0].upper() != "A"],
        "B": [w for w in word_list if w[0].upper() != "B"],
        "C": [w for w in word_list if w[0].upper() != "C"],
        "D": [w for w in word_list if w[0].upper() != "D"],
        "E": [w for w in word_list if w[0].upper() != "E"],
        "F": [w for w in word_list if w[0].upper() != "F"],
        "G": [w for w in word_list if w[0].upper() != "G"],
        "H": [w for w in word_list if w[0].upper() != "H"],
        "I": [w for w in word_list if w[0].upper() != "I"],
        "J": [w for w in word_list if w[0].upper() != "J"],
        "K": [w for w in word_list if w[0].upper() != "K"],
        "L": [w for w in word_list if w[0].upper() != "L"],
        "M": [w for w in word_list if w[0].upper() != "M"],
        "N": [w for w in word_list if w[0].upper() != "N"],
        "O": [w for w in word_list if w[0].upper() != "O"],
        "P": [w for w in word_list if w[0].upper() != "P"],
        "Q": [w for w in word_list if w[0].upper() != "Q"],
        "R": [w for w in word_list if w[0].upper() != "R"],
        "S": [w for w in word_list if w[0].upper() != "S"],
        "T": [w for w in word_list if w[0].upper() != "T"],
        "U": [w for w in word_list if w[0].upper() != "U"],
        "V": [w for w in word_list if w[0].upper() != "V"],
        "W": [w for w in word_list if w[0].upper() != "W"],
        "X": [w for w in word_list if w[0].upper() != "X"],
        "Y": [w for w in word_list if w[0].upper() != "Y"],
        "Z": [w for w in word_list if w[0].upper() != "Z"]
    }
    log_console_header("Total Words to Process", len(word_list))
    
    args_list = []
    for word1 in word_list:
        filtered_words = filter_word_list.get(word1[0].upper(), [])
        word1_pron = p_dict[word1][0]
        args_list.append((word1, word1_pron, filtered_words, p_dict, p_distance_dict))

    log_console_header("Starting Parallel Processing of Word Distances (this might take a minute)...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_word1_distances, args_list), total=len(args_list), desc="Calculating Word Averages", unit=" word"))

    log_console_header("Writing Word Averages to 'word_score_averages.csv'")
    with open("word_score_averages.csv", "w", newline="") as avg_csvfile:
        avg_writer = csv.writer(avg_csvfile)
        avg_writer.writerow(["Word", "Phonemes", "Magnitude", "Average Levenshtein Distance", "Average Phoneme Distance (Basic)", "Average Shared Phoneme Sequence Count"])
        
        for ind, args in tqdm(enumerate(args_list), desc="Processing Results", total=len(args_list), unit="word"):
            word1, word1_pron = args[:2]
            magnitude, total_dist_levenshtein, total_dist_phoneme, total_shared, count = results[ind]
            avg_levenshtein = total_dist_levenshtein / count
            avg_phoneme = total_dist_phoneme / count
            avg_shared = total_shared / count
            avg_writer.writerow([word1, " ".join(word1_pron), magnitude, avg_levenshtein, avg_phoneme, avg_shared])
            logging.debug(
                "Word: %-12s | Phonemes: %-20s | Magnitude: %.6f | Avg Levenshtein: %.6f | Avg Phoneme Dist: %.6f | Avg Shared Seq Count: %.6f",
                word1, " ".join(word1_pron), magnitude, avg_levenshtein, avg_phoneme, avg_shared
            )

    log_console_header("Word Averages Written to 'word_score_averages.csv'")

# --------------------------------
# Write the distance matrix for a given type (levenshtein, phoneme, shared).
# Creates a CSV file for each letter pair (e.g., A-B, A-C, etc.)
# NOTE: This function does not create redundant letter pairs
# - E.G., A-B and B-A are not created separately since they would be identical (just reversed).
# --------------------------------
def write_distance_matrix_csv(p_dict, p_distance_dict, words_by_letters):
    
    csv_base_dir = os.path.join("CSV Files")
    gephi_base_dir = os.path.join("Gephi Files")
    
    os.makedirs(csv_base_dir, exist_ok=True)
    os.makedirs(gephi_base_dir, exist_ok=True)
    
    header = ["Source", "Target", "Levenshtein Distance", "Phoneme Distance", "Shared Sequence Count", "Score"]
    letters = list(string.ascii_uppercase)

    # Start the main loop to create letter pairs
    for l1 in tqdm(letters, total=len(letters), desc=f"Calculating Distances...", unit=" letter", leave=False, colour="green"):
        
        # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        l1_index = letters.index(l1)
        l2_letters = letters[l1_index:]  
        
        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Calculating {l1}-Letter Pairs", unit=" letter pair", leave=False, colour="red"):  # Only create pairs (A-B, A-C, ..., B-C, ..., Z-Z)
            csv_filename = os.path.join(csv_base_dir, FILENAME_CSV_TEMPLATE.format(l1, l2))  
            gephi_filename = os.path.join(gephi_base_dir, FILENAME_CSV_TEMPLATE.format(l1, l2))    
            
            if os.path.exists(csv_filename) and os.path.exists(gephi_filename):
                continue
            
            word_list1 = words_by_letters[l1]
            word_list2 = words_by_letters[l2]

            # Generate all unique word pairs (w1, w2) where w1 is from word_list1 and w2 is from word_list2
            # We make sure w1 < w2 (alphabetized) to avoid duplicate pairs like (word1, word2) and (word2, word1)
            pairs = [(w1, w2) for w1 in word_list1 for w2 in word_list2 if w1 < w2]

            # Calculate every word pair's data and Write the results to CSV file for this letter pair
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
                with open(gephi_filename, "w", newline="") as gephi_file:
                    gephi_writer = csv.writer(gephi_file)
                    gephi_writer.writerow(["Source", "Target", "Weight"])   # Gephi header

                    # Calculate distances for each word pair
                    for w1, w2 in tqdm(pairs, total=len(pairs), desc=f"Calculating and Writing to CSV: {l1}_{l2}", unit=" word pair", leave=False, colour="yellow"):
                        lev = levenshtein_distance(w1, w2)
                        phon = phoneme_distance(p_dict[w1][0], p_dict[w2][0], p_distance_dict)
                        shared = shared_phoneme_sequences(p_dict[w1][0], p_dict[w2][0], p_distance_dict)
                        score = (lev * LEV_WEIGHT) + (phon * PHONEME_WEIGHT) - (shared * SHARED_WEIGHT)                        
                        
                        # Write the data to the main CSV file
                        writer.writerow([w1, w2, lev, phon, shared, score])
                        
                        # Write the data to the Gephi CSV file
                        gephi_writer.writerow([w1, w2, score])

# Write the average distances for each letter pair to a CSV file.
# This function reads all the CSV files in the "CSV Files" directory and calculates the averages for each letter pair.
# The results are written to a new CSV file named "letter_pair_averages.csv".
def write_distance_averages_csv():
    letters = list(string.ascii_uppercase)
    csv_base_dir = os.path.join("CSV Files")
    avg_csv_filename = os.path.join(csv_base_dir, "letter_pair_averages.csv")
    with open(avg_csv_filename, "w", newline="") as avg_file:
        avg_writer = csv.writer(avg_file)
        avg_writer.writerow(["Source", "Target", "Total Word Pairs",
                            "Min Lev", "Max Lev", "Avg Lev", "Std Dev Lev",
                            "Min Phoneme", "Max Phoneme", "Avg Phoneme", "Std Dev Phoneme",
                            "Min Shared", "Max Shared", "Avg Shared", "Std Dev Shared",
                            "Min Score", "Max Score", "Avg Score", "Std Dev Score"])
    
    if not os.path.exists(csv_base_dir):
        logging.warning(f"CSV base directory '{csv_base_dir}' does not exist. Skipping average calculation.")
        return

    for l1 in tqdm(letters, total=len(letters), desc="Calculating Letter Pair Averages", unit=" letter", colour="green"):
        l1_index = letters.index(l1)
        l2_letters = letters[l1_index:]  # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Calculating {l1}-Letter Pair Averages", unit=" letter pair", leave=False, colour="yellow"):
            data = {
                "levenshtein": [],
                "phoneme": [],
                "shared": [],
                "score": []
            }
            
            word_pairs = read_distance_matrix(l1, l2)
            for w1 in tqdm(word_pairs, total=len(word_pairs), desc=f"Processing {l1}-{l2} Word Pairs", unit=" word", leave=False, colour="red"):
                for w2 in word_pairs[w1]:
                    lev, phoneme, shared, score = word_pairs[w1][w2]
                    data["levenshtein"].append(lev)
                    data["phoneme"].append(phoneme)
                    data["shared"].append(shared)
                    data["score"].append(score)
            
            # Write the Letter-Pair Averages and other data points to the Letter-Pair Averages CSV file
            with open(avg_csv_filename, "a", newline="") as avg_file:
                avg_writer = csv.writer(avg_file)
                avg_writer.writerow([l1, l2, len(data["levenshtein"]),
                                    np.min(data["levenshtein"]),    np.max(data["levenshtein"]),    np.mean(data["levenshtein"]),   np.std(data["levenshtein"]),
                                    np.min(data["phoneme"]),        np.max(data["phoneme"]),        np.mean(data["phoneme"]),       np.std(data["phoneme"]),
                                    np.min(data["shared"]),         np.max(data["shared"]),         np.mean(data["shared"]),        np.std(data["shared"]),
                                    np.min(data["score"]),          np.max(data["score"]),          np.mean(data["score"]),         np.std(data["score"])])

# Read a distance matrix from a CSV file and return it as a dictionary
def write_distance_matrix_json():    
    json_base_dir = os.path.join("JSON Files")
    os.makedirs(json_base_dir, exist_ok=True)
    
    letters = list(string.ascii_uppercase)
    for l1 in tqdm(letters, total=len(letters), desc="Writing JSON Files", unit=" letter", colour="green"):
        l1_index = letters.index(l1)
        l2_letters = letters[l1_index:]  # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Writing JSON for {l1}-Letter Pairs", unit=" letter pair", leave=False, colour="red"):
            distance_dict = read_distance_matrix(l1, l2)
            if not distance_dict:
                logging.warning(f"No distance data found for {l1}-{l2}. Skipping JSON write.")
                continue

            json_filename = os.path.join("JSON Files", FILENAME_CSV_TEMPLATE.format(l1, l2).replace(".csv", ".json"))
            with open(json_filename, "w") as json_file:
                json.dump(distance_dict, json_file, indent=4)

def find_best_set_simulated_annealing(
    p_dict,
    p_distance_dict,
    words_by_letter,
    initial_temp=1000,
    cooling_rate=0.995,
    iterations=10000
):
    log_console_header("Starting Simulated Annealing Search", f"Iterations: {iterations}, Initial Temp: {initial_temp}, Cooling Rate: {cooling_rate}")

    letters = list(string.ascii_uppercase)
    
    # Initialize with a random candidate (1 word per letter)
    current_candidate = [random.choice(words_by_letter[l]) for l in letters if words_by_letter[l]]
    current_score, _, _, _ = _score_candidate(current_candidate, p_dict, p_distance_dict)
    
    best_candidate = current_candidate[:]
    best_score = current_score

    temp = initial_temp

    for step in tqdm(range(iterations), desc="Simulated Annealing Progress", unit="iteration"):
        # Pick a random partition to swap
        partition_idx = random.randint(0, len(letters) - 1)
        letter = letters[partition_idx]
        if not words_by_letter[letter]:
            continue  # Skip if no words for that letter
        new_word = random.choice(words_by_letter[letter])
        if new_word == current_candidate[partition_idx]:
            continue  # Skip if the new word is the same as current

        # Create new candidate by swapping one word
        new_candidate = current_candidate[:]
        new_candidate[partition_idx] = new_word

        new_score, _, _, _ = _score_candidate(new_candidate, p_dict, p_distance_dict)
        delta_score = new_score - current_score

        # Accept if better or probabilistically if worse
        if delta_score > 0 or math.exp(delta_score / temp) > random.random():
            current_candidate = new_candidate
            current_score = new_score
            if new_score > best_score:
                best_candidate = new_candidate
                best_score = new_score

        temp *= cooling_rate  # Cool down

    log_console_header("Simulated Annealing Completed")
    log_scores("Simulated Annealing Best Score", (best_score, best_candidate), p_dict)

    return best_candidate, best_score


# Read the letter_pair_averages.csv file and write the averages to a JSON file
def write_distance_averages_json():
    logging.info("Writing letter pair averages to JSON file")
    
    json_base_dir = os.path.join("JSON Files")
    os.makedirs(json_base_dir, exist_ok=True)
    csv_base_dir = os.path.join("CSV Files")
    avg_csv_filename = os.path.join(csv_base_dir, "letter_pair_averages.csv")
    if not os.path.exists(avg_csv_filename):
        logging.warning(f"Average CSV file '{avg_csv_filename}' does not exist. Skipping JSON write.")
        return
    
    averages_dict = defaultdict(dict)
    with open(avg_csv_filename, "r", newline="") as avg_file:
        reader = csv.reader(avg_file)
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            l1, l2, lev_avg, phoneme_avg, shared_avg, score_avg = row[0], row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5])
            averages_dict[l1][l2] = {
                "levenshtein": lev_avg,
                "phoneme": phoneme_avg,
                "shared": shared_avg,
                "score": score_avg
            }
            
    json_filename = os.path.join(csv_base_dir, "letter_pair_averages.json")
    with open(json_filename, "w") as json_file:
        json.dump(averages_dict, json_file, indent=4)

# ------------------------------------
# Read a distance matrix from a CSV file and return it as a dictionary.
# This function reads a distance matrix for a specific letter pair (e.g., A-B) and returns it as a dictionary.
# ------------------------------------
def read_distance_matrix(target_letter, compare_letter):
    # Read a distance matrix from a CSV file and return it as a dictionary
    filename = os.path.join("CSV Files", FILENAME_CSV_TEMPLATE.format(target_letter, compare_letter))
    distance_dict = {}
    
    if not os.path.exists(filename):
        logging.warning(f"Distance matrix file '{filename}' does not exist. Skipping read.")
        return distance_dict
    
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) < 3:
                continue
            word1, word2, lev, phoneme, shared, score = row[0], row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5])
            distance = (lev, phoneme, shared, score)
            if word1 not in distance_dict:
                distance_dict[word1] = {}
            distance_dict[word1][word2] = distance
    return distance_dict

# -------------------------------
# Dictionary Functions
# -------------------------------

# Compare every phoneme coordinate with every other, as well as itself and an empty string

def get_phoneme_distance_dict():
    phonemes = list(PHONEME_COORDINATES.keys())
    distance = defaultdict(float)
    for i, p1 in enumerate(phonemes):
        coord1 = PHONEME_COORDINATES[p1]
        distance[(p1, "")] = math.sqrt(sum(a ** 2 for a in coord1))
        distance[("", p1)] = math.sqrt(sum(a ** 2 for a in coord1))
        distance[(p1, p1)] = 0.0
        
        for j, p2 in enumerate(phonemes):
            if i < j:  # Avoid duplicate pairs
                coord2 = PHONEME_COORDINATES[p2]
                distance[(p1, p2)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                distance[(p2, p1)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                
    log_console_header("Phoneme Distance Dictionary Created | Total Phonemes", len(PHONEME_COORDINATES))
    
    # Print out the phoneme distance dictionary
    #logging.debug("Phoneme Distance Dictionary: %s", distance)
    
    if os.path.exists("phoneme_distance_dict.csv"):
        os.remove("phoneme_distance_dict.csv")
        
    log_console_header("Saving Phoneme Distance Dictionary to 'phoneme_distance_dict.csv'")
    # Save the phoneme distance dictionary to a file
    with open("phoneme_distance_dict.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phoneme 1", "Phoneme 2", "Distance"])
        for (p1, p2), dist in distance.items():
            writer.writerow([p1, p2, dist])
            
    logging.info("Phoneme Distance Dictionary Saved to 'phoneme_distance_dict.csv'")
    
    # Return the distance dictionary 
    return distance

# Clean the CMU Pronouncing Dictionary and apply filters.
def get_cleaned_cmu_dict():
    cleaned_dict = cmudict.dict()
    
    # Normalize the dictionary
    cleaned_dict = {
        word: prons for word, prons in tqdm(cleaned_dict.items(), desc="Cleaning CMU Dictionary", unit="word")
        if (
            (word in WORD_WHITELIST) or
            (word.isalpha()) and
            (word not in WORD_BLACKLIST) and
            (not any(word.startswith(prefix) for prefix in PREFIX_FILTERS)) and
            (len(prons[0]) >= MIN_PHONEME_LENGTH) and
            (len(prons[0]) <= MAX_PHONEME_LENGTH) and
            (len(word) >= MIN_WORD_LENGTH) and
            (len(word) <= MAX_WORD_LENGTH) and
            (len(set(word)) > 1) and
            (not any(word.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(word[0].lower(), []))) and
            (wordnet.synsets(word)) and
            (MIN_SYLLABLES <= len([p for p in prons[0] if p[-1].isdigit()]) <= MAX_SYLLABLES)
        )
    }              

    lemmatizer = WordNetLemmatizer()
    lemma_map = defaultdict(list)
    
    # Lemmatize the words, group by lemma
    for word, prons in tqdm(cleaned_dict.items(), desc="Lemmatizing Words", unit="word"):       
        noun = lemmatizer.lemmatize(word.lower(), pos='n')
        lemma = lemmatizer.lemmatize(noun, pos='v')
        lemma_map[lemma].append((word, prons[0]))

    temp_dict = {}
    # Choose representative word for each lemma (shortest spelling)
    for lemma, variants in lemma_map.items():
        representative = min(variants, key=lambda x: len(x[0]))
        temp_dict[representative[0]] = [representative[1]]  # keep as list for CMU compatibility 
    
    cleaned_dict = temp_dict

    # Normalize phoneme representation
    for word, pronunciations in tqdm(cleaned_dict.items(), desc="Normalizing Phonemes", unit="word"):
        cleaned_dict[word] = [[p.upper() for p in [p[:-1] if p[-1].isdigit() else p for p in pron]] for pron in pronunciations]

    # Add Custom Words to the dictionary
    for word, pron in CUSTOM_WORDS.items():
        cleaned_dict[word.lower()] = [pron]

    logging.info("CMU Pronouncing Dictionary Cleaned | Total Words: %d", len(cleaned_dict))

    return cleaned_dict

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

# Log a header message to the console
def log_console_header(message, data=None):
    logging.info("--------------------------------")
    if data:
        logging.info("%s: %s", message, data)
    else:
        logging.info(message)
    logging.info("--------------------------------")

# Log the best scores and words in a formatted way
def log_scores(list_name, set, p_dict):
    logging.info("--------------------------------")
    logging.info("Best %s Set (%.6f):", list_name, set[0])
    logging.info("--------------------------------")
    for word in set[1]:
        logging.info("%-12s  ->  %s", word.capitalize(), ' '.join(p_dict[word][0]))

# --------------------------------
# Graphing Functions
# --------------------------------

def plot_graph(spring_factor = 0.15, iters = 50, max_distance = 2.0):
    
    # Build the graph with networkx for easy Plotly integration
    G = nx.Graph()
    base_dir = os.path.join("Gephi Files")
    
    logging.info("Building Graph from CSV Files in '%s'", base_dir)
    logging.info("Spring Factor: %.2f | Iterations: %i | Score Range: %.2f", spring_factor, iters, max_distance)
    
    words = set()
    letters = string.ascii_uppercase
    
    for l1 in tqdm(letters, total=26, desc="Reading Word Distances", unit=" letter", colour="green"):
        
        if G.number_of_nodes() > 10000:
            break
        
        l1_index = letters.index(l1)
        check_list = letters[l1_index:]  # Only check letters after the current letter to avoid duplicates

        for l2 in tqdm(check_list, total=len(check_list), desc=f"- Reading {l1}-Word Distances | {len(G.nodes)} Nodes | {len(G.edges)} Edges", unit=" letter", leave=False, colour="yellow"):
            filename = os.path.join(base_dir, FILENAME_CSV_TEMPLATE.format(l1, l2))
            edges = defaultdict(list)

            # Sort the csv file by distance before processing
            with open(filename, newline="") as f:
                reader = csv.reader(f)
                next(reader) # Skip header

                # Process the sorted rows
                for w1, w2, dist in tqdm(reader, desc=f"Processing {l1}-{l2} Distances", unit=" word pair", leave=False, colour="red"):
                    if w1 != w2 and np.abs(float(dist)) <= max_distance:
                        words.add(w1)
                        words.add(w2)
                        edges[dist].append((w1, w2))
                    
                    if len(words) > 10000:
                        break
            
                # Add nodes and edges to the graph
                # Doing this in the for loop so we can see progress and avoid memory issues with large graphs
                G.add_nodes_from(tqdm(words, total=len(words), desc="Adding Nodes", unit="word", leave=False))
                for d in tqdm(edges.keys(), total=len(edges), desc="Processing Edges", unit="distance", leave=False):
                    G.add_edges_from(edges[d], weight=float(d))
    
    logging.info("Total Words: %d | Total Edges: %d", G.number_of_nodes(), G.number_of_edges())

    # Assign colors by first letter
    letters = sorted(set(w[0].upper() for w in words))
    letter_to_color = {l: f"hsl({int(360*i/len(letters))},70%,50%)" for i, l in enumerate(letters)}
    node_colors = [letter_to_color[w[0].upper()] for w in G.nodes()]
    
    log_console_header("Calculating Node Positions for Scatter Plot (This will take a while...)")
    logging.info("Using spring layout with factor: %.2f and iterations: %d", spring_factor, iters)
    # Use spring layout for positions
    pos = nx.spring_layout(G, k=spring_factor, iterations=iters)

    # Build edge traces
    edge_x = edge_y = []
    for e in tqdm(G.edges(), total=len(G.edges()), desc="Building Edge Traces", unit="edge"):
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    log_console_header("Calculating Edge Traces for Scatter Plot (This will take a minute...)")
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgb(50, 50, 50)"),
        hoverinfo='none',
        mode='lines'
    )

    # Build node traces
    node_x = []
    node_y = []
    node_text = []
    for node in tqdm(G.nodes(), total=len(G.nodes()), desc="Building Node Traces", unit="node"):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=8,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Cluster Graph: Iterations: {iters}, Score Range: {max_distance}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    #fig.write_image(f"cluster_graph_iter_{iters}_score_range_{max_distance}.png", width=1200, height=800, scale=2)
    fig.write_html(f"cluster_graph_iter_{iters}_score_range_{max_distance}.html")
    fig.show()

def plot_a_word_levenshtein_bargraph():
    
    letters = sorted(string.ascii_uppercase)
    averages = {l: [] for l in letters}
    
    if os.path.exists("CSV Files/Levenshtein Distance Matrices/levenshtein_averages.csv"):
        averages = defaultdict(list)
        with open("CSV Files/Levenshtein Distance Matrices/levenshtein_averages.csv", newline="") as avg_file:
            avg_reader = csv.reader(avg_file)
            next(avg_reader)
            for row in avg_reader:
                target_letter, _, avg_distance = row
                averages[target_letter].append(float(avg_distance))
                    
    # If the averages file does not exist, calculate the averages
    else:
        with open("CSV Files/Levenshtein Distance Matrices/levenshtein_averages.csv", "w", newline="") as avg_file:
            avg_writer = csv.writer(avg_file)
            avg_writer.writerow(["Target Letter", "Compared Letter", "Average Levenshtein Distance"])

        for l1 in string.ascii_uppercase:
            for l2 in tqdm(string.ascii_uppercase, total = 26, desc=f"Calculating {l1}-Word Distances", unit=" letter"):
                filename = f"CSV Files/Levenshtein Distance Matrices/levenshtein_matrix_{l1}_{l2}.csv"  
                distances = []
                with open(filename, newline="") as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    for row in reader:
                        _, _, dist = row
                        distances.append(float(dist))

                    if distances:
                        averages[l1].append(np.mean(distances))

                    # Write the average to a file
                    with open("CSV Files/Levenshtein Distance Matrices/levenshtein_averages.csv", "a", newline="") as avg_file:
                        avg_writer = csv.writer(avg_file)
                        avg_writer.writerow([l1, l2, np.mean(distances) if distances else 0.0])

    x_labels = np.arange(len(letters))
    width = 1.0/(52.0)
    multiplier = 0.0
    
    fig, ax = plt.subplots(layout = "constrained")
    
    for attr, mean in averages.items():
        offset = width * multiplier
        rects = ax.bar(x_labels + offset, mean, width=width, label=f"[{attr}]", color=matplotlib.cm.tab20(multiplier / 26.0))
        ax.bar_label(rects, padding=2, fmt='%.4f', fontsize=6, color='black', rotation=90, label_type='edge')
        multiplier += 1.0
    
    ax.set_title("Average Levenshtein Distance: [A]-Letter vs. [X]-Letter Words")
    ax.set_xlabel("Target Letter")
    ax.set_ylabel("Average Levenshtein Distance")
    
    ax.set_xticks(x_labels + width, letters)
    ax.set_xticklabels(letters, rotation=45)
    ax.set_ylim(min(min(averages[l]) for l in letters) - 0.05, max(max(averages[l]) for l in letters) + 0.25)
    ax.legend(loc='lower left', ncols=7, title="Target Letter")
    
    #plt.tight_layout()
    plt.savefig("levenshtein_bargraph.png")
    plt.show()

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------

# -----------------------------
# Randomization Options
# -----------------------------
TRIALS = 10000                   # Number of random trials
PENALIZE_LENGTH_VARIANCE = True # Set to 'True' to penalize length variance
LENGTH_VARIANCE_WEIGHT = 10     # Adjust this weight to control penalty severity

# -----------------------------
# Scoring Weights
# -----------------------------
LEV_WEIGHT = 1.0
PHONEME_WEIGHT = 1.0
SHARED_WEIGHT = 1.0

# -----------------------------
# Graphing Options
# -----------------------------
SPRING_FACTOR = 0.5    # Factor to control the spring force in the graph layout
ITERATIONS = 50         # Number of iterations for the spring layout algorithm
MAX_DISTANCE = 3.0        # Maximum distance between nodes in the graph layout

def main():
    
    log_console_header("Loading and Cleaning CMU Dictionary")

    # Load the CMU Pronouncing Dictionary and create the phoneme distance dictionary
    nltk.download('cmudict')
    nltk.download('wordnet')

    PHONEME_DICT = get_cleaned_cmu_dict()
    PHONEME_DISTANCE_DICT = get_phoneme_distance_dict()
    WORDS_BY_LETTER = defaultdict(list)
    for word in PHONEME_DICT.keys():
        WORDS_BY_LETTER[word[0].upper()].append(word)

    log_console_header("Best Phonetic Alphabet Utility")
    print("Choose an option:")
    print("1. Generate Word Averages")
    print("2. Generate Distance Matrices")
    print("3. Find Best Randomized Trial")
    print("4. Plot Graph")
    print("5. Plot Levenshtein Bar Graph")
    print("6. Generate Custom Phonetic Alphabet")
    print("7. Find Best Simulated Annealing Trial")
    print("Q. Quit")
    choice = input("Enter your choice (1/2/3/4/5/Q): ").strip().lower()

    if choice == '1':
        log_console_header("Generating Word Averages")
        write_word_averages(PHONEME_DICT, PHONEME_DISTANCE_DICT)
    elif choice == '2':
        print("\nWARNING: This operation may take a long time and will generate a large number of files where you downloaded the repo!")
        print("Press Enter to continue or Ctrl+C to cancel.")
        input()
        print("---------------------------------")
        print("Generate CSV or JSON files for each letter pair?")
        print("1. CSV")
        print("2. JSON (Requires CSV files to be generated first)")
        print("3. Both CSV and then JSON")
        matrix_choice = input("Enter your choice (1/2): ").strip()
        if matrix_choice == '1':
            log_console_header("Generating CSV Distance Matrices")
            write_distance_matrix_csv(PHONEME_DICT, PHONEME_DISTANCE_DICT, WORDS_BY_LETTER)
            log_console_header("Writing Distance Averages to CSV")
            write_distance_averages_csv()
        elif matrix_choice == '2':
            log_console_header("Generating JSON Distance Matrices")
            write_distance_matrix_json()
            log_console_header("Writing Distance Averages to JSON")
            write_distance_averages_json()
        elif matrix_choice == '3':
            log_console_header("Generating CSV Distance Matrices")
            write_distance_matrix_csv(PHONEME_DICT, PHONEME_DISTANCE_DICT, WORDS_BY_LETTER)
            log_console_header("Writing Distance Averages to CSV")
            write_distance_averages_csv()
            log_console_header("Generating JSON Distance Matrices")
            write_distance_matrix_json()
            log_console_header("Writing Distance Averages to JSON")
            write_distance_averages_json()
        else:
            print("Invalid choice. Please run the program again.")
            return
    elif choice == '3':
        log_console_header("Finding Best Randomized Trial")
        best_scores = find_best_set_randomized(PHONEME_DICT, PHONEME_DISTANCE_DICT, WORDS_BY_LETTER, TRIALS)
        # Log best sets
        log_scores("Levenshtein Distance", best_scores['levenshtein'], PHONEME_DICT)
        log_scores("Phoneme Distance", best_scores['phoneme'], PHONEME_DICT)
        log_scores("Shared Phoneme Sequence Count", best_scores['shared'], PHONEME_DICT)
        log_scores("Overall", best_scores['score'], PHONEME_DICT)
    elif choice == '4':
        log_console_header("Plotting Cluster Graph")
        plot_graph(spring_factor=SPRING_FACTOR, iters=ITERATIONS, max_distance=MAX_DISTANCE)
    elif choice == '5':
        log_console_header("Plotting Levenshtein Bar Graph")
        plot_a_word_levenshtein_bargraph()
    elif choice == '6':
        # Ask the user for a list of words
        user_words = input("Enter a list of words separated by commas: ").strip().split(',')
        user_words = [word.strip().lower() for word in user_words if word.strip()]
        
        # For testing, generate a random list of words from the CMU dictionary
        custom_dict = random.sample(list(PHONEME_DICT.keys()), random.randint(1, 26))
        # custom_dict = {word: PHONEME_DICT[word] for word in user_words if word in PHONEME_DICT}
        if not custom_dict:
            print("No valid words found in the CMU dictionary. Please run the program again.")
            return
        find_best_set_with_preselected(custom_dict, PHONEME_DICT, PHONEME_DISTANCE_DICT, WORDS_BY_LETTER, TRIALS)
    elif choice == '7':
        log_console_header("Finding Best Simulated Annealing Trial")
        best_candidate, best_score = find_best_set_simulated_annealing(PHONEME_DICT, PHONEME_DISTANCE_DICT, WORDS_BY_LETTER, iterations=100000)
        log_scores("Simulated Annealing", (best_score, best_candidate), PHONEME_DICT)
    elif choice == 'q':
        log_console_header("Exiting Program")
        return
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()