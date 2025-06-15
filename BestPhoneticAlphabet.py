import os
import csv
import math
import random
import logging
import string
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures

import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

import editdistance

import plotly.graph_objects as go
import networkx as nx

import pandas as pd
import seaborn as sns
from collections import Counter

from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import librosa
import soundfile as sf
from fastdtw import fastdtw
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram

import warnings

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

# NOTE: Temporary fix, should properly download and integrate ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv*", category=RuntimeWarning)

WORD_PAIR_FILENAME_TEMPLATE = "word_pairs_{0}_{1}_data.csv"
LETTER_PAIR_FILENAME = "letter_pair_averages.csv"
CSV_DISTANCE_HEADERS = ["Source", "Target", "Score",
        "Source-Phonemes", "Target-Phonemes",
        "Levenshtein Distance", "Phoneme Distance", 
        "Shared Sequence Count", "Sequences",
        "Suffix Count", "Suffixes",
        "Rhyme Count", "Rhymes"]
CSV_DISTANCE_AVERAGE_HEADERS = ["Source", "Target", "Total Word Pairs",
                            "Min Lev", "Max Lev", "Avg Lev", "Std Dev Lev",
                            "Min Phoneme", "Max Phoneme", "Avg Phoneme", "Std Dev Phoneme",
                            "Min Shared", "Max Shared", "Avg Shared", "Std Dev Shared",
                            "Min Score", "Max Score", "Avg Score", "Std Dev Score"]

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
    "m" : ["N"],
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

PHONEME_DICT = defaultdict(list)
PHONEME_DISTANCE_DICT = defaultdict()
PHONEME_AUDIO_DISTANCE_DICT = defaultdict()
WORDS_BY_LETTER = defaultdict(list)

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

"""_summary_
Extract the rhyme portion of a phoneme list.
NOTE: Requires the phoneme list to have stress markers (e.g., '1' for primary stress).
"""
def extract_rhyme_portion(phonemes):
    rhyme = []
    found_primary_stress = False
    for ph in phonemes:
        if found_primary_stress:
            rhyme.append(ph)
        elif ph[-1] == '1':  # look for primary stress
            found_primary_stress = True
            rhyme.append(ph)
    return tuple(rhyme) if found_primary_stress else tuple()

"""_summary_
Normalize a phoneme by removing stress digits.
This is useful for comparing phonemes without stress.
"""
def normalize_phoneme(p_list):
    return [p[:-1] if p[-1].isdigit() else p for p in p_list]

"""_summary_
Generate candidate words based on input letters and preselected words.
"""
def candidate_gen(trials, letters, words_by_letter, preselected_by_letter=None):
    for _ in range(trials):
        candidate = []
        for l in letters:
            if l in preselected_by_letter:
                candidate.append(preselected_by_letter[l])
            elif words_by_letter[l]:
                candidate.append(random.choice(words_by_letter[l]))
        if len(candidate) == len(letters):
            yield candidate

"""_summary_
Update the best scores dictionary with a new candidate's scores.
"""
def update_bests(best_scores, score, lev, phon, seq, suffix, c):
    best_scores["levenshtein"] = max(best_scores["levenshtein"], (lev, c), key=lambda x: x[0])
    best_scores["phoneme"] = max(best_scores["phoneme"], (phon, c), key=lambda x: x[0])
    best_scores["seq"] = min(best_scores["seq"], (seq, c), key=lambda x: x[0])
    best_scores["suffix"] = min(best_scores["suffix"], (suffix, c), key=lambda x: x[0])
    best_scores["score"] = max(best_scores["score"], (score, c), key=lambda x: x[0])
    return best_scores

"""_summary_
Helper function to unpack arguments for multiprocessing.
"""
def _score_candidate_unpack(args):
    return _score_candidate(*args)

"""_summary_
Helper function for scoring a candidate.
Scores a candidate set of words using:
    (
        Total Normalized Levenshtein distance
        + Normalized phoneme distance
        - Shared sequence penalty (phoneme subarrays)
        - Shared suffix penalty (orthographic + phonemic)
        - Rhyme penalty
    ) x 100 for scaling
"""
def _score_candidate(selected_words, p_dict, p_distance_dict, p_audio_dist_dict, phoneme_suffix_length=2):
    total_levenshtein = total_phoneme_distance = total_phoneme_audio_dist = 0
    shared_sequence_penalty = shared_suffix_penalty = 0
    rhyme_penalty = 0

    vowels = [v for v in PHONEME_COORDINATES if PHONEME_COORDINATES[v][0] == 0]
    consonants = [c for c in PHONEME_COORDINATES if PHONEME_COORDINATES[c][0] == 1]

    vowel_set = set()
    consonant_set = set()
    suffix_counts = {}
    suffix_to_words = {}
    phoneme_suffix_counts = {}
    phoneme_suffix_to_words = {}
    shared_sequences_list = []
    shared_suffixes_list = []
    rhyme_pairs = []

    # Pairwise comparisons
    for i in range(len(selected_words)):
        w1 = selected_words[i]
        p1 = normalize_phoneme(p_dict[w1][0])   # Normalize the phoneme list (remove stress markers)
        
        # Collect vowels and consonants
        for p in p1:
            if p in vowels:
                vowel_set.add(p)
            if p in consonants:
                consonant_set.add(p)

        # Collect orthographic suffixes (2-5 characters)
        word_len = len(w1)
        for slen in range(2, min(6, word_len + 1)):
            suffix = w1[-slen:]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            suffix_to_words.setdefault(suffix, []).append(w1)

        # Collect phoneme suffixes
        if len(p1) >= phoneme_suffix_length:
            phon_suffix = tuple(p1[-phoneme_suffix_length:])
            phoneme_suffix_counts[phon_suffix] = phoneme_suffix_counts.get(phon_suffix, 0) + 1
            phoneme_suffix_to_words.setdefault(phon_suffix, []).append(w1)
        
        for j in range(i + 1, len(selected_words)):
            w2 = selected_words[j]
            p2 = normalize_phoneme(p_dict[w2][0])
            
            # Rhyme penalty (using CMU dictionary with stress)
            # NOTE: Requires UN-NORMALIZED Phonemes (i.e., with stress markers)
            rhyme1 = extract_rhyme_portion(p_dict[w1][0])
            rhyme2 = extract_rhyme_portion(p_dict[w2][0])
            if rhyme1 and (rhyme1 == rhyme2):
                rhyme_penalty += 1
                rhyme_pairs.append((w1, w2, rhyme1))

            total_levenshtein += editdistance.eval(w1, w2)
            total_phoneme_distance += sum(p_distance_dict.get((ph1, ph2), 1) for ph1, ph2 in zip(p1, p2))
            total_phoneme_audio_dist += sum(p_audio_dist_dict.get((ph1, ph2), 1) for ph1, ph2 in zip(p1, p2))

            # Shared phoneme sub-sequences (n-grams)
            ngram_max = min(len(p1), len(p2))
            for n in range(1, ngram_max + 1):
                ngrams1 = {tuple(p1[k:k+n]) for k in range(len(p1) - n + 1)}
                ngrams2 = {tuple(p2[k:k+n]) for k in range(len(p2) - n + 1)}
                shared_ngrams = ngrams1.intersection(ngrams2)
                for ngram in shared_ngrams:
                    shared_sequence_penalty += 1
                    shared_sequences_list.append((w1, w2, ngram))

    # Orthographic suffixes
    for suffix, count in suffix_counts.items():
        if count > 1:
            shared_suffix_penalty += (count - 1)
            shared_suffixes_list.append((f"orthographic: {suffix}", suffix_to_words[suffix]))

    # Phoneme suffixes
    for suffix, count in phoneme_suffix_counts.items():
        if count > 1:
            shared_suffix_penalty += (count - 1)
            shared_suffixes_list.append((f"phonemic: {suffix}", phoneme_suffix_to_words[suffix]))

    # Normalizations
    lev_norm = (total_levenshtein - 1.0) / 9.0  # Normalized to [0-1] scale (1 is best, 0 is worst) [9.0 is avg]
    phon_norm = total_phoneme_distance / 15.68  # Normalized to [0-1] scale (1 is best, 0 is worst) [15.68 avg]
    audio_norm = total_phoneme_audio_dist / 1000  # Normalized to [0-1] scale (1 is best, 0 is worst) [1000 avg] # NOTE: Should get the value it's being divided by
    seq_norm = shared_sequence_penalty / 36.0   # 36 is max pairs in 9 phonemes
    suffix_norm = shared_suffix_penalty / 36.0  # 36 is max pairs in 9 phonemes
    rhyme_norm = rhyme_penalty / 25.0           # 25 is max pairs in 26 words
    vowel_norm = len(vowel_set) / len(vowels)   # Vowel diversity bonus/penalty [0-1 scale]
    consonant_norm = len(consonant_set) / len(consonants)   # Consonant diversity bonus/penalty [0-1 scale]

    # Weights
    WEIGHT_LEVENSHTEIN = 1.0
    WEIGHT_PHONEME = 1.0
    WEIGHT_SHARED_SEQ = 2.0
    WEIGHT_SHARED_SUFFIX = 2.0
    WEIGHT_RHYME = 2.0
    WEIGHT_VOWEL_DIVERSITY = 1.0
    WEIGHT_CONSONANT_DIVERSITY = 1.0
    WEIGHT_AUDIO_DIVERSITY = 1.0

    score = (
        (WEIGHT_LEVENSHTEIN * lev_norm)
        + (WEIGHT_PHONEME * phon_norm)
        - (WEIGHT_SHARED_SEQ * seq_norm)
        - (WEIGHT_SHARED_SUFFIX * suffix_norm)
        - (WEIGHT_RHYME * rhyme_norm)
        + (WEIGHT_VOWEL_DIVERSITY * vowel_norm)
        + (WEIGHT_CONSONANT_DIVERSITY * consonant_norm)
        + (WEIGHT_AUDIO_DIVERSITY * audio_norm)
    ) * 100.0

    return {
        "score": score,
        "total_levenshtein": total_levenshtein,
        "total_phoneme_distance": total_phoneme_distance,
        "shared_sequence": (shared_sequence_penalty, shared_sequences_list),
        "shared_suffix": (shared_suffix_penalty, shared_suffixes_list),
        "rhyme": (rhyme_penalty, rhyme_pairs),
        "vowel_diversity": vowel_norm
    }

"""_summary_
Find the best set of words via random sampling.
Randomized search for best phonetic alphabet, allowing for preselected words.
preselected_words: list of words to lock in for their starting letter (case-insensitive).
"""
def find_best_set_randomized(p_dict, p_distance_dict, p_audio_dist_dict, words_by_letter, trials=1000, batch_size=1000, preselected_words=None):
    log_console_header("Starting Randomized Search for Best Set of Words", trials)
    best_scores = {
        "levenshtein": (float('-inf'), []),
        "phoneme": (float('-inf'), []),
        "seq": (float('inf'), []),
        "suffix": (float('inf'), []),
        "score": (float('-inf'), []),
    }

    letters = list(string.ascii_uppercase)

    # Handle preselected words
    preselected_by_letter = {}
    if preselected_words:
        for w in preselected_words:
            l = w[0].upper()
            preselected_by_letter[l] = min(preselected_by_letter[l], w) if l in preselected_by_letter else w

    headers = ["Score", "Total Levenshtein Distance", "Total Phoneme Distance", 
            "Total Shared Sequences", "Total Shared Suffixes"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with open("random_search_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        batch = []
        for c in tqdm(candidate_gen(trials, letters, words_by_letter, preselected_by_letter), total=trials, desc="Generating and Scoring Candidates", unit="candidate"):
            batch.append(c)
            if len(batch) >= batch_size:
                # Build a minimal p_dict for this batch
                all_words = set(w for candidate in batch for w in candidate)
                mini_p_dict = {w: p_dict[w] for w in all_words}
                # Prepare arguments for each candidate
                args = [(candidate, mini_p_dict, p_distance_dict, p_audio_dist_dict) for candidate in batch]
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = list(tqdm(
                        executor.map(_score_candidate_unpack, args),
                        total=len(args),
                        desc=f"Scoring Candidates in Batch | Current Best Score: {best_scores['score'][0]}",
                        unit="candidate", leave=False
                    ))
                for c, data in zip(batch, results):
                    writer.writerow([data["score"], 
                                    data["total_levenshtein"],
                                    data["total_phoneme_distance"],
                                    data["shared_sequence"][0],
                                    data["shared_suffix"][0]] 
                                    + sorted(list(c)))
                    best_scores = update_bests(best_scores, 
                                                data["score"], 
                                                data["total_levenshtein"], 
                                                data["total_phoneme_distance"],
                                                data["shared_sequence"][0],
                                                data["shared_suffix"][0], 
                                                c)
                batch = []

    return best_scores

"""_summary_
Write the distance matrix for a given type (levenshtein, phoneme, shared).
Creates a CSV file for each letter pair (e.g., A-B, A-C, etc.)
NOTE: This function does not create redundant letter pairs
- E.G., A-B and B-A are not created separately since they would be identical (just reversed).
"""
def write_distance_matrix_csv(p_dict, p_distance_dict, p_audio_dist_dict, words_by_letters):

    csv_base_dir = os.path.join("CSV Files")
    gephi_base_dir = os.path.join("Gephi Files")
    
    os.makedirs(csv_base_dir, exist_ok=True)
    os.makedirs(gephi_base_dir, exist_ok=True)
    
    letters = list(string.ascii_uppercase)

    # Start the main loop to create letter pairs
    for l1 in tqdm(letters, total=len(letters), desc="Calculating Distances...", unit=" letter", leave=False, colour="green"):
        
        # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        l1_index = letters.index(l1)
        l2_letters = letters[l1_index + 1:] if l1_index + 1 < len(letters) else []  
        
        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Calculating {l1}-Letter Pairs", unit=" letter pair", leave=False, colour="red"):  # Only create pairs (A-B, A-C, ..., B-C, ..., Z-Z)
            csv_filename = os.path.join(csv_base_dir, WORD_PAIR_FILENAME_TEMPLATE.format(l1, l2))  
            gephi_filename = os.path.join(gephi_base_dir, WORD_PAIR_FILENAME_TEMPLATE.format(l1, l2))    
            
            word_list1 = words_by_letters[l1]
            word_list2 = words_by_letters[l2]

            # Generate all unique word pairs (w1, w2) where w1 is from word_list1 and w2 is from word_list2
            # We make sure w1 < w2 (alphabetized) to avoid duplicate pairs like (word1, word2) and (word2, word1)
            pairs = [(w1, w2) for w1 in word_list1 for w2 in word_list2 if w1 < w2]

            # Calculate every word pair's data and Write the results to CSV file for this letter pair
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_DISTANCE_HEADERS)
                
                with open(gephi_filename, "w", newline="") as gephi_file:
                    gephi_writer = csv.writer(gephi_file)
                    gephi_writer.writerow(["Source", "Target", "Weight"])   # Gephi header

                    # Calculate distances for each word pair
                    for w1, w2 in tqdm(pairs, total=len(pairs), desc=f"Calculating and Writing to CSV: {l1}_{l2}", unit=" word pair", leave=False, colour="yellow"):
                        data = _score_candidate([w1, w2], p_dict, p_distance_dict, p_audio_dist_dict)                  
                        p1, p2 = p_dict[w1][0], p_dict[w2][0]
                        # Write the data to the main CSV file   
                        writer.writerow([w1, w2, data["score"],
                                        p1, p2,
                                        data["total_levenshtein"], data["total_phoneme_distance"],
                                        data["shared_sequence"][0], data["shared_sequence"][1],
                                        data["shared_suffix"][0], data["shared_suffix"][1],
                                        data["rhyme"][0], data["rhyme"][1]])   
                    
                        gephi_writer.writerow([w1, w2, data["score"]])  # Write the data to the Gephi CSV file

"""_summary_
Write the average distances for each letter pair to a CSV file.
This function reads all the CSV files in the "CSV Files" directory and calculates the averages for each letter pair.
The results are written to a new CSV file named "letter_pair_averages.csv".
"""
def write_distance_averages_csv():
    
    letters = list(string.ascii_uppercase)
    avg_filename = os.path.join("CSV Files", LETTER_PAIR_FILENAME)
    
    with open(avg_filename, "w", newline="") as avg_file:
        avg_writer = csv.writer(avg_file)
        avg_writer.writerow(CSV_DISTANCE_AVERAGE_HEADERS)

    for l1 in tqdm(letters, total=len(letters), desc="Calculating Letter Pair Averages", unit=" letter", colour="green"):
        l1_index = letters.index(l1)
        # Get letters after l1 (not including l1 itself) This avoids redundant pairs like A-B and B-A
        l2_letters = letters[l1_index + 1:] if l1_index + 1 < len(letters) else []
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
                    data["levenshtein"].append(word_pairs[w1][w2]["levenshtein"])
                    data["phoneme"].append(word_pairs[w1][w2]["phoneme"])
                    data["shared"].append(word_pairs[w1][w2]["shared"])
                    data["score"].append(word_pairs[w1][w2]["score"])
            
            # Write the Letter-Pair Averages and other data points to the Letter-Pair Averages CSV file
            with open(avg_filename, "a", newline="") as avg_file:
                    avg_writer = csv.writer(avg_file)
                    l_min, l_max, l_avg, l_std = np.min(data["levenshtein"]),    np.max(data["levenshtein"]),    np.mean(data["levenshtein"]),   np.std(data["levenshtein"])
                    p_min, p_max, p_avg, p_std = np.min(data["phoneme"]),        np.max(data["phoneme"]),        np.mean(data["phoneme"]),       np.std(data["phoneme"])
                    s_min, s_max, s_avg, s_std = np.min(data["shared"]),         np.max(data["shared"]),         np.mean(data["shared"]),        np.std(data["shared"])
                    x_min, x_max, x_avg, x_std = np.min(data["score"]),          np.max(data["score"]),          np.mean(data["score"]),         np.std(data["score"])
                    
                    avg_writer.writerow([l1, l2, len(data["levenshtein"]),
                                    l_min, l_max, l_avg, l_std,
                                    p_min, p_max, p_avg, p_std,
                                    s_min, s_max, s_avg, s_std,
                                    x_min, x_max, x_avg, x_std])

"""_summary_
Exports a Gephi-compatible .gexf graph file where:
- Nodes are words grouped by first letter
- Edge weights are inverse of dissimilarity scores
- Also stores raw dissimilarity score per edge
"""
def export_scored_graph_to_gexf(word_groups, p_dict, p_distance_dict, p_audio_dist_dict, filename="Gephi Files/scored_graph.gexf", max_per_group=100, epsilon=1e-6):
    G = nx.Graph()

    # Limit size per group
    trimmed_groups = { letter: words[:max_per_group]for letter, words in word_groups.items() if words }

    # Add nodes with group labels
    for letter, words in tqdm(trimmed_groups.items(), total=len(trimmed_groups), desc="Adding nodes"):
        for word in tqdm(words, total=len(words), desc=f"Adding {letter} Words", leave=False):
            G.add_node(word, group=letter)

    # Add edges with inverse score weights
    letters = list(trimmed_groups.keys())
    for i in tqdm(range(len(letters)), desc="Calculating Edges", total=len(letters)):
        for j in tqdm(range(i + 1, len(letters)), desc=f"On {letters[i]}", total=len(letters)-1, leave=False):
            for w1 in tqdm(trimmed_groups[letters[i]], total=len(trimmed_groups[letters[i]]), desc=f"On {letters[i]}_{letters[j]} Letter Word Pairs...", leave=False):
                for w2 in trimmed_groups[letters[j]]:
                    try:
                        c = [w1, w2]
                        score = _score_candidate(c, p_dict, p_distance_dict, p_audio_dist_dict)["score"]
                        weight = max(epsilon, 1 / (score + epsilon))
                        G.add_edge(w1, w2, raw_score=score, weight=weight)
                    except Exception as e:
                        print(f"⚠️ Error scoring pair ({w1}, {w2}): {e}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    log_console_header(f"Writing Gephi graph to {filename}...")
    nx.write_gexf(G, filename)
    print(f"✅ Exported {len(G.nodes)} nodes and {len(G.edges)} edges to '{filename}'")

"""_summary_
Read a distance matrix from a CSV file and return it as a dictionary.
This function reads a distance matrix for a specific letter pair (e.g., A-B) and returns it as a dictionary.
"""
def read_distance_matrix(target_letter, compare_letter):
    filename = os.path.join("CSV Files", WORD_PAIR_FILENAME_TEMPLATE.format(target_letter, compare_letter))
    distance_dict = defaultdict(dict)
    if not os.path.exists(filename):
        logging.warning(f"Distance matrix file '{filename}' does not exist.")
        return distance_dict
    
    with open(filename, "r", newline="") as f:
        data = defaultdict()
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in tqdm(reader, desc=f"Reading {filename}...", colour="blue", leave=False, unit=" row"):
            
            # Create a dictionary for the row data based on the CSV headers
            for h in CSV_DISTANCE_HEADERS:
                data[h] = row[CSV_DISTANCE_HEADERS.index(h)]
                
            word1, word2 = data["Source"], data["Target"]
            if word1 not in distance_dict:
                distance_dict[word1] = {}
                
            distance_dict[word1][word2] = {
                "levenshtein": float(data["Levenshtein Distance"]),
                "phoneme": float(data["Phoneme Distance"]),
                "shared": int(data["Shared Sequence Count"]),
                "score": float(data["Score"])
            }

    return distance_dict

"""_summary_
Synthesize a word's audio by concatenating its phoneme audio files.
"""
def synthesize_words(p_dict):
    for word in tqdm(p_dict, total=len(p_dict), desc="Synthesizing Words", unit=" word"):     
        p_list = normalize_phoneme(p_dict[word][0])
        synthesize_word_audio(word, p_list)

# -------------------------------
# Dictionary Functions
# -------------------------------

"""_summary_
Compare every phoneme coordinate with every other, as well as itself and an empty string
"""
def get_phoneme_distance_dict():

    if os.path.exists("phoneme_distance_dict.csv"):
        logging.info("Phoneme Distance Dictionary already exists. Loading from 'phoneme_distance_dict.csv'.")
        distance = defaultdict(float)
        with open("phoneme_distance_dict.csv", "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                p1, p2, dist = row
                distance[(p1, p2)] = distance[(p2, p1)] = float(dist)
        return distance
    
    phonemes = list(PHONEME_COORDINATES.keys())
    distance = defaultdict(float)
    for i, p1 in enumerate(phonemes):
        coord1 = PHONEME_COORDINATES[p1]
        distance[(p1, "")] = math.sqrt(sum (a ** 2 for a in coord1))
        distance[("", p1)] = math.sqrt(sum (a ** 2 for a in coord1))
        distance[(p1, p1)] = 0.0

        for j, p2 in enumerate(phonemes):
            if i < j:  # Avoid duplicate pairs
                coord2 = PHONEME_COORDINATES[p2]
                distance[(p1, p2)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                distance[(p2, p1)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    log_console_header("Phoneme Distance Dictionary Created | Total Phonemes", len(PHONEME_COORDINATES))
    # Save the phoneme distance dictionary to a file
    with open("phoneme_distance_dict.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phoneme 1", "Phoneme 2", "Distance"])
        for (p1, p2), dist in tqdm(distance.items(), desc="Writing Phoneme Distance Dictionary", unit=" entry"):
            writer.writerow([p1, p2, dist])
    logging.info("Phoneme Distance Dictionary Saved to 'phoneme_distance_dict.csv'")
    
    # Return the distance dictionary 
    return distance

"""_summary_
Computes the audio differences between phonemes based on their waveform representations.
"""
def get_phoneme_audio_difference_dict(base_folder="Phoneme Voice Files", target_sr=22050, phoneme_file_template="_normalized.wav"):
    dist_matrix = defaultdict(float)

    if os.path.exists("phoneme_audio_difference_matrix.csv"):
        logging.info("Phoneme Audio Distance Matrix already exists. Loading from 'phoneme_audio_difference_matrix.csv'.")
        with open("phoneme_audio_difference_matrix.csv", "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                p1, p2, dist = row
                dist_matrix[(p1, p2)] = dist_matrix[(p2, p1)] = float(dist)
        return dist_matrix

    phonemes = list(PHONEME_COORDINATES.keys())
    waveforms = {}

    for ph in phonemes:
        path = os.path.join(base_folder, ph, f"{ph}{phoneme_file_template}")
        if not os.path.exists(path):
            logging.warning("Medoid waveform not found for phoneme: %s", ph)
            continue
        y, _ = librosa.load(path, sr=target_sr)
        y = librosa.util.normalize(y)
        waveforms[ph] = y


    for i, p1 in tqdm(enumerate(waveforms.keys()), total=len(waveforms.keys()), desc="Computing Distances", unit=" phoneme"):
        dist_matrix[(p1, "")] = dist_matrix[("", p1)] = dist_matrix[(p1, p1)] = 0.0
        for j, p2 in tqdm(enumerate(waveforms.keys()), total=len(waveforms.keys()), desc=f"Computing distances for {p1}", unit=" phoneme", leave=False):
            if i != j:
                dist, _ = fastdtw(waveforms[p1], waveforms[p2], dist=lambda x, y: np.abs(x - y))
                dist_matrix[(p1, p2)] = dist_matrix[(p2, p1)] = dist

    with open("phoneme_audio_difference_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phoneme 1", "Phoneme 2", "Distance"])
        for (p1, p2), dist in tqdm(dist_matrix.items(), total=len(dist_matrix), desc="Writing Phoneme Distance Dictionary", unit=" entry"):
            writer.writerow([p1, p2, dist])

    return dist_matrix

"""_summary_
Clean the CMU Pronouncing Dictionary and apply filters.
- Filters out words with non-alphabetic characters, too short/long words, blacklisted words, etc.
- Lemmatizes words to their base forms and groups them by lemma, selecting the shortest variant.
- Adds custom words with predefined phonemes.
"""
def get_cleaned_cmu_dict():

    cleaned_dict = cmudict.dict()
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

    # Choose representative word for each lemma (shortest spelling)
    temp_dict = {}
    for lemma, variants in lemma_map.items():
        representative = min(variants, key=lambda x: len(x[0]))
        temp_dict[representative[0]] = [representative[1]]  # keep as list for CMU compatibility 
    
    cleaned_dict = temp_dict

    # Add Custom Words to the dictionary
    for word, pron in CUSTOM_WORDS.items():
        cleaned_dict[word.lower()] = [pron]

    logging.info("CMU Pronouncing Dictionary Cleaned | Total Words: %d", len(cleaned_dict))

    return cleaned_dict

"""_summary_
Groups words by their first letter, length, phoneme length, and syllable count.
Returns four dictionaries:
- words_by_letter
- words_by_length
- words_by_phoneme_length
- words_by_syllable
"""
def get_words_data(cmu_dict):
    words_by_letter = defaultdict(list)
    words_by_length = defaultdict(list)
    words_by_phoneme_length = defaultdict(list)
    words_by_syllable = defaultdict(list)
    
    # Populate the words_by_letter dictionary
    for word in tqdm(cmu_dict.keys(), desc="Grouping Words by First Letter", unit="word"):
        first_letter = word[0].upper()
        length = len(word)
        phoneme_length = len(cmu_dict[word][0])
        syllable_count = len([p for p in cmu_dict[word][0] if p[-1].isdigit()])  # Count syllables based on stress markers

        # Add the word to the appropriate lists
        # Only add words that start with an alphabetic character
        if first_letter.isalpha():
            words_by_letter[first_letter].append(word)
        words_by_length[length].append(word)
        words_by_phoneme_length[phoneme_length].append(word)
        words_by_syllable[syllable_count].append(word)

    # Sort each letter's word list alphabetically
    for letter in words_by_letter:
        words_by_letter[letter].sort()

    logging.info("Words Data Collected | Words grouped by Letter, Length, Phoneme Length, and Syllable")

    return words_by_letter, words_by_length, words_by_phoneme_length, words_by_syllable

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

"""_summary_
Log a header message to the console
"""
def log_console_header(message, data=None):
    logging.info("--------------------------------")
    if data:
        logging.info("%s: %s", message, data)
    else:
        logging.info(message)
    logging.info("--------------------------------")

"""_summary
Log the best scores and words in a particular format.
"""
def log_scores(list_name, set, p_dict):
    logging.info("--------------------------------")
    logging.info("Best %s Set (%.6f):", list_name, set[0])
    logging.info("--------------------------------")
    for word in set[1]:
        logging.info("%-12s  ->  %s", word.capitalize(), ' '.join(p_dict[word][0]))

# ----------------------------
# AUDIO ANALYSIS FUNCTIONS
# ----------------------------

"""_summary_
Estimates when an audio clip is "silent"
"""
def estimate_silence_threshold(audio, sample_length_ms=100):
    samples = [audio[i:i+sample_length_ms].dBFS for i in range(0, len(audio), sample_length_ms)]
    quiet_samples = [s for s in samples if s != float('-inf')]
    return min(quiet_samples) - 5 if quiet_samples else -40

"""_summary_
Normalize the audio chunk to the target dBs.
"""
def normalize_to_target(chunk, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - chunk.dBFS
    return chunk.apply_gain(change_in_dBFS)

"""_summary_
Attempts to split up the main phoneme recording sessions into individual phoneme audio clips
"""
def segment_phoneme_sessions(folder_path="Phoneme Voice Files", min_silence_len=10):
    for filename in tqdm(os.listdir(folder_path), desc="Processing Phoneme Audio Files", unit="file", colour="blue"):
        if not filename.lower().endswith(".wav"):
            continue

        phoneme = os.path.splitext(filename)[0].upper()
        file_path = os.path.join(folder_path, filename)

        try:
            audio = AudioSegment.from_wav(file_path)
            silence_thresh = estimate_silence_threshold(audio)

            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=-40,
                keep_silence=5,
                seek_step=1
            )

            if not chunks:
                print(f"⚠️ No utterances detected for {phoneme}")
                continue

            out_dir = os.path.join(folder_path, phoneme)
            os.makedirs(out_dir, exist_ok=True)

            for i, chunk in tqdm(enumerate(chunks, 1), desc=f"Segmenting {phoneme} | Auto Silence Threshold {silence_thresh:.2f} dBFS", unit="segment", leave=False, colour="green"):
                normalized_chunk = normalize_to_target(chunk)
                out_path = os.path.join(out_dir, f"{phoneme}_{i:02d}.wav")
                normalized_chunk.export(out_path, format="wav")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

"""_summary_
For each phoneme folder inside base_folder:
- Loads all utterances
- Finds the utterance with the lowest total DTW distance to the others
- Saves it as {phoneme}_medoid.wav
"""
def find_medoid_phoneme_recordings(base_folder="Phoneme Voice Files", target_sr=22050, output_filename="medoid.wav"):
    for phoneme in tqdm(sorted(os.listdir(base_folder)), desc="Finding Medoid Phoneme Recordings", unit="phoneme", colour="blue"):
        folder = os.path.join(base_folder, phoneme)
        if not os.path.isdir(folder):
            continue

        wav_files = sorted([f for f in os.listdir(folder) if 
                            f.endswith(".wav") and not f.endswith(output_filename) and not f.endswith("average.wav") and not f.endswith("median.wav")])

        if not wav_files:
            print(f"⚠️ No WAV files found in {folder}")
            continue

        waveforms = []
        paths = []

        for f in tqdm(wav_files, desc=f"Loading {phoneme} Recordings", unit="file", leave=False, colour="green"):
            path = os.path.join(folder, f)
            y, _ = librosa.load(path, sr=target_sr)
            y = librosa.util.normalize(y)
            waveforms.append(y)
            paths.append(path)

        n = len(waveforms)
        distances = np.zeros((n, n))

        # Compute pairwise DTW distances
        for i in range(n):
            for j in range(i + 1, n):
                dist, _ = fastdtw(waveforms[i], waveforms[j], dist=lambda x, y: np.abs(x - y))
                distances[i, j] = distances[j, i] = dist

        # Sum distances for each waveform
        totals = distances.sum(axis=1)
        best_idx = np.argmin(totals)

        # Save the medoid waveform as output
        best_wave = waveforms[best_idx]
        sf.write(os.path.join(folder, phoneme + "_" + output_filename), best_wave, target_sr)

"""_summary_
Resamples each phoneme's medoid.wav to match a fixed duration (default: 0.2 seconds).
Saves result as normalized.wav in the same folder.
"""
def normalize_phoneme_durations(base_folder="Phoneme Voice Files", target_duration=0.15, target_sr=22050, input_filename="_medoid.wav", output_filename="_normalized.wav"):
    for phoneme in PHONEME_COORDINATES.keys():
        folder = os.path.join(base_folder, phoneme)
        in_path = os.path.join(folder, f"{phoneme}{input_filename}")
        out_path = os.path.join(folder, f"{phoneme}{output_filename}")

        if not os.path.isfile(in_path):
            print(f"- Missing {phoneme}{input_filename} for {phoneme}")
            continue

        y, sr = librosa.load(in_path, sr=target_sr)
        duration = librosa.get_duration(y=y, sr=sr)
        stretch_factor = duration / target_duration

        # Skip if already within ~5% of target duration
        if 0.95 <= stretch_factor <= 1.05:
            sf.write(out_path, y, sr)
            continue

        # Time-stretch to match target duration
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        sf.write(out_path, y_stretched, sr)

"""_summary_
Synthesizes a word audio file by concatenating medoid phoneme .wav files.
Parameters:
- word: the word string (e.g. "apple")
- phonemes: list of ARPAbet phonemes (e.g. ["AE", "P", "AH", "L"])
- phoneme_folder: where medoid.wav files are stored (subfolders by phoneme)
- output_base: base folder to write word .wav file into
"""
def synthesize_word_audio(word, phoneme_list, phoneme_folder="Phoneme Voice Files", output_base="Word Voicings", phoneme_file_suffix="_normalized.wav", crossfade_ms=20, envelope_ms=10):
    # Ensure output directory exists
    letter = word[0].upper()
    output_dir = os.path.join(output_base, letter)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{word}.wav")

    # Combine phoneme audio files
    combined = None
    for ph in phoneme_list:
        ph_folder = os.path.join(phoneme_folder, ph)
        ph_file = os.path.join(ph_folder, f"{ph}{phoneme_file_suffix}")
        if not os.path.exists(ph_file):
            print(f"Warning: Phoneme audio file not found: {ph_file}")
            continue
        ph_audio = AudioSegment.from_wav(ph_file)
        ph_audio = ph_audio.fade_in(envelope_ms).fade_out(envelope_ms)
        if combined is None:
            combined = ph_audio
        else:
            combined = combined.append(ph_audio, crossfade=crossfade_ms)

    # Export the combined audio
    # Normalize the final audio for consistent volume
    combined = effects.normalize(combined)
    combined.export(output_path, format="wav")
    return output_path

# -----------------------------
# Visualization Functions
# -----------------------------

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
            filename = os.path.join(base_dir, WORD_PAIR_FILENAME_TEMPLATE.format(l1, l2))
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

"""_summary_
Plots 4 bar graphs:
1. Number of words in each letter group (A-Z)
2. Number of words for each word length
3. Number of words for each phoneme length
4. Number of words for each syllable count
"""
def plot_dictionary_stats(words_by_letter, words_by_length, words_by_phoneme_length, words_by_syllable):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. Words per letter group
    letters = sorted(words_by_letter.keys())
    counts = [len(words_by_letter[l]) for l in letters]
    bars = axs[0, 0].bar(letters, counts, color='tab:blue')
    axs[0, 0].set_title("Number of Words by First Letter")
    axs[0, 0].set_xlabel("First Letter")
    axs[0, 0].set_ylabel("Word Count")
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        axs[0, 0].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 2. Words per word length
    lengths = sorted(words_by_length.keys())
    length_counts = [len(words_by_length[l]) for l in lengths]
    bars = axs[0, 1].bar(lengths, length_counts, color='tab:orange')
    axs[0, 1].set_title("Number of Words by Word Length")
    axs[0, 1].set_xlabel("Word Length")
    axs[0, 1].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[0, 1].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 3. Words per phoneme length
    phoneme_lengths = sorted(words_by_phoneme_length.keys())
    phoneme_counts = [len(words_by_phoneme_length[l]) for l in phoneme_lengths]
    bars = axs[1, 0].bar(phoneme_lengths, phoneme_counts, color='tab:green')
    axs[1, 0].set_title("Number of Words by Phoneme Length")
    axs[1, 0].set_xlabel("Phoneme Length")
    axs[1, 0].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[1, 0].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # 4. Words per syllable count
    syllable_counts = sorted(words_by_syllable.keys())
    syllable_word_counts = [len(words_by_syllable[s]) for s in syllable_counts]
    bars = axs[1, 1].bar(syllable_counts, syllable_word_counts, color='tab:red')
    axs[1, 1].set_title("Number of Words by Syllable Count")
    axs[1, 1].set_xlabel("Syllable Count")
    axs[1, 1].set_ylabel("Word Count")
    for bar in bars:
        height = bar.get_height()
        axs[1, 1].annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.suptitle("Dictionary Statistics", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dictionary_stats.png")
    plt.show()  

def visualize_word_length_distribution(words_by_letter):
    """
    Visualize the distribution of word lengths for each letter group in words_by_letter.
    Shows both a heatmap and a grouped bar chart.
    """
    # Count word lengths for each letter
    length_counts_by_letter = {}
    for letter, words in words_by_letter.items():
        lengths = [len(word) for word in words]
        length_counts_by_letter[letter] = Counter(lengths)

    # Find all unique word lengths
    all_lengths = set()
    for counts in length_counts_by_letter.values():
        all_lengths.update(counts.keys())
    all_lengths = sorted(all_lengths)

    # Build DataFrame: rows=word lengths, columns=letters
    df = pd.DataFrame(
        {letter: [length_counts_by_letter[letter].get(l, 0) for l in all_lengths] for letter in words_by_letter},
        index=all_lengths
    )
    df.index.name = "Word Length"

    # Plot heatmap
    plt.figure(figsize=(16, 6))
    sns.heatmap(df.T, cmap="Blues", annot=True, fmt="d")
    plt.title("Word Length Distribution by Letter (Heatmap)")
    plt.xlabel("Word Length")
    plt.ylabel("Starting Letter")
    plt.tight_layout()
    plt.show()

    # Plot grouped bar chart
    df.T.plot(kind="bar", stacked=False, figsize=(16, 6))
    plt.title("Word Length Distribution by Letter (Bar Chart)")
    plt.xlabel("Starting Letter")
    plt.ylabel("Count")
    plt.legend(title="Word Length")
    plt.tight_layout()
    plt.show()

def plot_phoneme_distance_matrix(matrix, labels, title="Phoneme Difference Matrix"):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="magma", square=True)
    plt.title(title)
    plt.xlabel("Phoneme")
    plt.ylabel("Phoneme")
    plt.tight_layout()
    plt.show()

def plot_phoneme_dendrogram(matrix, labels):
    linkage_matrix = linkage(matrix, method='average')
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title("Phoneme Clustering Dendrogram")
    plt.ylabel("Acoustic Distance")
    plt.tight_layout()
    plt.show()

def plot_mds(matrix, labels, dim=2):
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(matrix)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = coords[i][:2]
        plt.scatter(x, y)
        plt.text(x, y, label, fontsize=10)
    plt.title("MDS Projection of Phonemes")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid()
    plt.tight_layout()
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
    
    normalize_phoneme_durations()  # Ensure phoneme audio files are normalized to target duration
    PHONEME_AUDIO_DISTANCE_DICT = get_phoneme_audio_difference_dict("Phoneme Voice Files")
    WORDS_BY_LETTER, WORDS_BY_LENGTH, WORDS_BY_PHONEME_LENGTH, WORDS_BY_SYLLABLE = get_words_data(PHONEME_DICT)

    choices = {
        '1': "Generate Distance Matrices",
        '2': "Generate Letter Pair Averages",
        '3': "Find Best Randomized Trial",
        '4': "Plot Graph",
        '5': "Plot Dictionary Stats",
        '6': "Plot Stats 2",
        '7': "Create Gephi File",
        '8': "Split Phoneme Audio Takes",
        '9': "Create Average Phoneme Audio Files",
        '10': "Create and Plot Phoneme Audio Distance",
        '11': "Score Premade Alphabet",
        '12': "Synthesize Word Audio"
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

    if choice == "Generate Distance Matrices":
        log_console_header("Generating Distance Matrices")
        print("\n---------------WARNING-----------------")
        print("\nThis operation will take a long time and will generate a LARGE number of BIG files!!")
        print("\nPress Enter to continue or Ctrl+C to cancel.")
        input()
        print("---------------------------------")
        log_console_header("Generating CSV Distance Matrices")
        write_distance_matrix_csv(PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT, WORDS_BY_LETTER)
    elif choice == "Generate Letter Pair Averages":
        write_distance_averages_csv()
    elif choice == "Find Best Randomized Trial":
        log_console_header("Finding Best Phonetic Alphabet via Randomized Trial")
        best_scores = find_best_set_randomized(PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT, WORDS_BY_LETTER, TRIALS)
        # Log best sets
        log_scores("Levenshtein Distance", best_scores['levenshtein'], PHONEME_DICT)
        log_scores("Phoneme Distance", best_scores['phoneme'], PHONEME_DICT)
        log_scores("Shared Phoneme Sequence Count", best_scores['seq'], PHONEME_DICT)
        log_scores("Overall", best_scores['score'], PHONEME_DICT)
    elif choice == "Plot Graph":
        log_console_header("Plotting Cluster Graph")
        plot_graph(spring_factor=SPRING_FACTOR, iters=ITERATIONS, max_distance=MAX_DISTANCE)
    elif choice == "Plot Dictionary Stats":
        log_console_header("Plotting Dictionary Stats")
        plot_dictionary_stats(WORDS_BY_LETTER, WORDS_BY_LENGTH, WORDS_BY_PHONEME_LENGTH, WORDS_BY_SYLLABLE)
    elif choice == "Plot Stats 2":
        visualize_word_length_distribution(WORDS_BY_LETTER)
    elif choice == "Create Gephi File":
        export_scored_graph_to_gexf(WORDS_BY_LETTER, PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT, max_per_group=100)
    elif choice == "Split Phoneme Audio Takes":
        segment_phoneme_sessions()
    elif choice == "Create Average Phoneme Audio Files":
        find_medoid_phoneme_recordings()
    elif choice == "Create and Plot Phoneme Audio Distance":
        matrix, labels = get_phoneme_audio_difference_dict("Phoneme Voice Files")
        plot_phoneme_distance_matrix(matrix, labels)
        plot_phoneme_dendrogram(matrix, labels)
        plot_mds(matrix, labels)
    elif choice == "Score Premade Alphabet":
        NATO = [w for w in NATO_PHONETIC_ALPHABET if w in PHONEME_DICT]
        best_scores = _score_candidate(NATO, PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT)
        logging.info("--------------------------------")
        logging.info(f"NATO Score ({best_scores['score']:,.2f}):")
        logging.info("--------------------------------")
        for word in NATO:
            logging.info("%-8s  ->  %s", word.capitalize(), ' '.join(PHONEME_DICT[word][0]))
    elif choice == "Synthesize Word Audio":
        synthesize_words(PHONEME_DICT)
    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()