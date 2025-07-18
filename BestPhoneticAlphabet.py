import os
import csv
import math
import random
import logging
import json
import string
import time
from collections import defaultdict
from turtle import distance

# Scientific and Data Libraries
import numpy as np
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

# Distance and Edit Libraries
import editdistance

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
CSV_DISTANCE_HEADERS:list[str] = [
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

"""
Words that we don't want in our alphabets
These are words that are either offensive, inappropriate, or just don't fit the criteria
Some other reasons for blacklisting words:
- Not caught by the filters
- Strange pronunciations that don't fit the phonetic alphabet
- Words that are too similar to other words in the alphabet
- Slurs and racial remarks are a no-go, we diverse and tolerant in this sum-bitch
"""
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

"""
Prefixes that should not be at the start of words
These prefixes are often silent or not pronounced, so we filter them out
"""
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

"""
Prefix filters for phonemes
Certain phonemes should not be at the start of words in certain letter groups
For example, "E" words shouldn't start with "y" like "eunuch" or "euphoria"
"""
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

PHONEME_DICT = defaultdict(list)
PHONEME_DICT_NORMALIZED = defaultdict(list)
PHONEME_DISTANCE_DICT = defaultdict()
PHONEME_AUDIO_DISTANCE_DICT = defaultdict()
PHONEME_COORD_MATRIX = None
PHONEME_AUDIO_MATRIX = None
PHONEME_MATRIX_INDEX = {}

WORDS_BY_LETTER = defaultdict(list)

# --------------------------------
# Calculation Functions
# TODO: Extract these to separate file/module probably maybw
# --------------------------------

def build_phoneme_distance_matrix(p_distance_dict):
    # 1. Find all unique phonemes
    all_phonemes = sorted(set(ph for pair in p_distance_dict for ph in pair))
    ph_to_idx = {ph: i for i, ph in enumerate(all_phonemes)}
    size = len(all_phonemes)

    # 2. Fill square distance matrix
    ph_dist_matrix = np.ones((size, size), dtype=np.float32)
    for (ph1, ph2), dist in p_distance_dict.items():
        i = ph_to_idx[ph1]
        j = ph_to_idx[ph2]
        ph_dist_matrix[i, j] = dist
        ph_dist_matrix[j, i] = dist  # symmetric

    return ph_dist_matrix, ph_to_idx

""" Extract the rhyme portion of a phoneme list.
NOTE: Requires the phoneme list to have stress markers (e.g., '1' for primary stress).
Probably doesn't really work, doh well
"""
# NOTE: Probably doesn't work as intended, doh
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

""" Remove stress digits from phonemes. """
def normalize_phoneme(p_list):
    return [p[:-1] if p[-1].isdigit() else p for p in p_list]

def candidate_gen(trials, words_by_letter, preselected_by_letter=None):
    for _ in range(trials):
        candidate = []
        for le in LETTERS:
            if le in preselected_by_letter:
                candidate.append(preselected_by_letter[le])
            elif words_by_letter[le]:
                candidate.append(random.choice(words_by_letter[le]))
        if len(candidate) == len(LETTERS):
            yield candidate

def _score_candidate(selected_words, p_dict, phoneme_suffix_length=2, weights=None):

    total_levenshtein, total_phoneme_coord_dist, total_phoneme_audio_dist = 0, 0, 0
    shared_sequence_penalty, shared_suffix_penalty = 0, 0
    rhyme_penalty = 0

    vowels = [v for v in PHONEME_COORDINATES if PHONEME_COORDINATES[v][0] == 0]
    consonants = [c for c in PHONEME_COORDINATES if PHONEME_COORDINATES[c][0] == 1]

    vowel_set, consonant_set = set(), set()
    suffix_counts, suffix_to_words = {}, {}
    phoneme_suffix_counts, phoneme_suffix_to_words = {}, {}
    shared_sequences_list, shared_suffixes_list = [], []
    rhyme_pairs = []
        
    # Pairwise comparisons
    for i in range(len(selected_words)):
        w1 = selected_words[i]
        p1 = PHONEME_DICT_NORMALIZED[w1]

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
        
        # Compare to all subsequent words
        for j in range(i + 1, len(selected_words)):
            w2 = selected_words[j]
            p2 = PHONEME_DICT_NORMALIZED[w2]

            total_levenshtein += editdistance.eval(w1, w2)
            
            # --- Vectorized phoneme COORD distance ---
            idx1 = np.array([PHONEME_MATRIX_INDEX.get(ph, -1) for ph in p1])
            idx2 = np.array([PHONEME_MATRIX_INDEX.get(ph, -1) for ph in p2])

            min_len = min(len(idx1), len(idx2))
            if min_len > 0:
                idx1 = idx1[:min_len]
                idx2 = idx2[:min_len]

                pairwise = PHONEME_COORD_MATRIX[idx1, idx2]
                pairwise = np.where((idx1 == -1) | (idx2 == -1), 1.0, pairwise)

                total_phoneme_coord_dist += np.sum(pairwise)

            # --- Vectorized phoneme AUDIO distance ---
            if min_len > 0:
                pairwise_audio = PHONEME_AUDIO_MATRIX[idx1, idx2]
                pairwise_audio = np.where((idx1 == -1) | (idx2 == -1), 1.0, pairwise_audio)

                total_phoneme_audio_dist += np.sum(pairwise_audio)


            # Rhyme penalty (using CMU dictionary with stress)
            #   NOTE: Requires UN-NORMALIZED Phonemes (i.e., with stress markers)
            #   TODO: Probably doesn't actually work as intended tbh....
            rhyme1 = extract_rhyme_portion(p_dict[w1][0])
            rhyme2 = extract_rhyme_portion(p_dict[w2][0])
            if rhyme1 and (rhyme1 == rhyme2):
                rhyme_penalty += 1
                rhyme_pairs.append((w1, w2, rhyme1))

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

    # Weights
    WEIGHT_LEVENSHTEIN =            weights["weight_levenshtein"] if weights is not None           else 1.0
    WEIGHT_PHONEME =                weights["weight_phoneme"] if weights is not None               else 1.0
    WEIGHT_SHARED_SEQ =             weights["weight_shared_seq"] if weights is not None            else 2.0
    WEIGHT_SHARED_SUFFIX =          weights["weight_shared_suffix"] if weights is not None         else 2.0
    WEIGHT_RHYME =                  weights["weight_rhyme"] if weights is not None                 else 2.0
    WEIGHT_VOWEL_DIVERSITY =        weights["weight_vowel_diversity"] if weights is not None       else 1.0
    WEIGHT_CONSONANT_DIVERSITY =    weights["weight_consonant_diversity"] if weights is not None   else 1.0
    WEIGHT_AUDIO_DIVERSITY =        weights["weight_audio_diversity"] if weights is not None       else 1.0

    score = (
        (WEIGHT_LEVENSHTEIN     * total_levenshtein)
        + (WEIGHT_PHONEME       * total_phoneme_coord_dist)
        + (WEIGHT_AUDIO_DIVERSITY * total_phoneme_audio_dist)
        + (WEIGHT_VOWEL_DIVERSITY       * (len(vowel_set) / len(vowels)))
        + (WEIGHT_CONSONANT_DIVERSITY   * (len(consonant_set) / len(consonants)))
        - (WEIGHT_SHARED_SEQ    * shared_sequence_penalty)
        - (WEIGHT_SHARED_SUFFIX * shared_suffix_penalty)
        - (WEIGHT_RHYME         * rhyme_penalty)
    )

    # Individual Candidate Scores totaled
    return {
        "score": score,
        "total_levenshtein": total_levenshtein,
        "total_phoneme_distance": total_phoneme_coord_dist,
        "phoneme_audio_distance": total_phoneme_audio_dist,
        "shared_sequence": (shared_sequence_penalty, shared_sequences_list),
        "shared_suffix": (shared_suffix_penalty, shared_suffixes_list),
        "rhyme": (rhyme_penalty, rhyme_pairs),
        "vowel_diversity": (len(vowel_set) / len(vowels)),
    }

# --------------
# Main Functions
# TODO: Extract these to a separate file/module
# --------------

""" Find the best set of words via random sampling.
Randomized search for best phonetic alphabet, allowing for preselected words.
"""
def find_best_set_randomized(p_dict, words_by_letter, trials=1000, preselected_words=None, top_candidates=100):

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
        results = _score_candidate(c, p_dict)
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

# --------------
# WRITE CSV FUNCTIONS
# TODO: Extract these to a separate file/module
# --------------

""" Writes the distance matrix for a given type (levenshtein, phoneme, shared).
Creates a CSV file for each letter pair (e.g., A-B, A-C, etc.)
NOTE: Does not create redundant letter pairs
- E.G., A-B and B-A are not created separately since they would be identical (just reversed).
"""
def write_word_pair_scores(p_dict, words_by_letters):

    csv_base_dir = os.path.join("CSV Files")
    gephi_base_dir = os.path.join("Gephi Files")
    
    os.makedirs(csv_base_dir, exist_ok=True)
    os.makedirs(gephi_base_dir, exist_ok=True)
    
    # Start the main loop to create letter pairs
    for l1 in tqdm(LETTERS, total=len(LETTERS), desc="Calculating Distances...", unit=" letter", leave=False, colour="green"):

        # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        l1_index = LETTERS.index(l1)
        l2_letters = LETTERS[l1_index + 1:] if l1_index + 1 < len(LETTERS) else []

        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Calculating {l1}-Letter Pairs", unit=" letter pair", leave=False, colour="red"):  # Only create pairs (A-B, A-C, ..., B-C, ..., Z-Z)
            csv_filename = os.path.join(csv_base_dir, WORD_PAIR_FILENAME_TEMPLATE.format(l1, l2))    
            
            word_list1 = words_by_letters[l1]
            word_list2 = words_by_letters[l2]

            # Generate all unique word pairs (w1, w2) where w1 is from word_list1 and w2 is from word_list2
            # We make sure w1 < w2 (alphabetized) to avoid duplicate pairs like (word1, word2) and (word2, word1)
            pairs = [(w1, w2) for w1 in word_list1 for w2 in word_list2 if w1 < w2]

            # Calculate every word pair's data and Write the results to CSV file for this letter pair
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_DISTANCE_HEADERS)
                
                # Calculate distances for each word pair
                for w1, w2 in tqdm(pairs, total=len(pairs), desc=f"Calculating and Writing to CSV: {l1}_{l2}", unit=" word pair", leave=False, colour="yellow"):
                    data = _score_candidate([w1, w2], p_dict)                  
                    p1, p2 = p_dict[w1][0], p_dict[w2][0]
                    # Write the data to the main CSV file   
                    writer.writerow([w1, w2, data["score"],
                                    p1, p2,
                                    data["total_levenshtein"], data["total_phoneme_distance"],
                                    data["shared_sequence"][0], data["shared_sequence"][1],
                                    data["shared_suffix"][0], data["shared_suffix"][1],
                                    data["rhyme"][0], data["rhyme"][1],
                                    data["phoneme_audio_distance"]])   

def write_word_averages(p_dict, words_by_letter):

    log_console_header("Calculating Word Averages")
    csv_filename = os.path.join("CSV Files", "word_averages.csv")

    # Create the CSV file and write the header. We'll write the results as we calculate
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)     
        writer.writerow(CSV_WORD_AVERAGE_HEADERS)

    # For loops written in a way so we don't constantly go
    #   "gimmie the dictionary with all the words that don't start with this letter" for each word
    # probably a better way to do this, bleh

    for letter1 in tqdm(LETTERS, desc="Processing Letters", unit="letter", colour="green", leave=False):
        word_averages = []
       
        for word in tqdm(words_by_letter[letter1], desc="Calculating word averages", unit="word", colour="yellow", leave=False):     
            scores = {
                "score": [],
                "total_levenshtein": [],
                "total_phoneme_distance": [],
                "shared_sequence": [],
                "shared_suffix": [],
                "rhyme": []
            }
            # Calculate phoneme magnitude for the word
            phonemes = PHONEME_DICT_NORMALIZED[word][0]
            phoneme_magnitude = 0.0
            for p in phonemes:
                if p in PHONEME_COORDINATES:
                    coords = PHONEME_COORDINATES[p]
                    magnitude = np.sqrt(sum(coord**2 for coord in coords))
                    phoneme_magnitude += magnitude

            # Calculate scores against all other words starting with different letters
            for letter2 in tqdm(LETTERS, desc=f"Comparing {word[0].upper()} with other letters", unit=" letter", leave=False, colour="red"):
                if letter2 == letter1:
                    continue
                for other_word in tqdm(words_by_letter[letter2], desc=f"Scoring {word}", unit="comparison", leave=False, colour="blue"):
                    result = _score_candidate([word, other_word], p_dict, p_distance_dict, p_audio_dist_dict, phoneme_suffix_length=2, weights=None)
                    scores["score"].append(result["score"])
                    scores["total_levenshtein"].append(result["total_levenshtein"])
                    scores["total_phoneme_distance"].append(result["total_phoneme_distance"])
                    scores["shared_sequence"].append(result["shared_sequence"][0])
                    scores["shared_suffix"].append(result["shared_suffix"][0])
                    scores["rhyme"].append(result["rhyme"][0])

            word_data = {
                'word': word,
                'letter': word[0].upper() if word else '',
                'phonemes': ' '.join(p_dict[word][0]),
                'phoneme_count': len(p_dict[word][0]),
                'phoneme_magnitude': phoneme_magnitude,
                'comparisons_made': sum(len(scores[k]) for k in scores),
                **{
                    f"{stat}_{metric if metric != 'score' else 'score' if stat != 'avg' else 'score'}": func(scores[metric])
                    for metric in scores
                    for stat, func in zip(['avg', 'min', 'max', 'std'], [np.mean, np.min, np.max, np.std])
                }
            }
            word_averages.append(word_data)

        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            for word_data in tqdm(word_averages, desc=f"Writing {letter1}-Word data", unit="word", colour="white", leave=False):
                row = [word_data.get(h.lower(), "") for h in CSV_WORD_AVERAGE_HEADERS]
                writer.writerow(row)

    
    log_console_header(f"Word averages saved to '{csv_filename}'")
    return word_averages

def write_phoneme_coord_distance_dict():
    # Sort PHONEME_COORDINATES by phoneme name for consistent ordering
    distances = defaultdict(float)
    sorted_phonemes = sorted(PHONEME_COORDINATES.keys())

    for p1 in tqdm(sorted_phonemes, desc="Calculating Phoneme Distances", unit=" phoneme", colour="green"):
        coord1 = PHONEME_COORDINATES[p1]
        
        # Calculate distance from origin (0, 0, 0, 0) to each phoneme coordinate
        distances[(p1, "")] = distances[("", p1)] = math.sqrt(sum(a ** 2 for a in coord1))
        distances[(p1, p1)] = 0.0

        for p2 in sorted_phonemes:
            if p1 < p2:
                coord2 = PHONEME_COORDINATES[p2]
                distances[(p1, p2)] = distances[(p2, p1)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    # Save the phoneme distance dictionary to a file
    with open(PHONEME_COORDINATE_DISTANCE_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phoneme 1", "Phoneme 2", "Distance"])
        for (p1, p2), dist in tqdm(distances.items(), desc="Writing Phoneme Distance Dictionary", unit=" entry"):
            writer.writerow([p1, p2, dist])
    logging.info(f"Phoneme Coordinate Distance Dictionary Saved to '{PHONEME_COORDINATE_DISTANCE_FILENAME}'")

    return distances

# -------------------------------
# READ CSV Functions
# TODO: Extract Read file functions into a separate module
# -------------------------------

""" Reads a csv file for a specific letter pair (e.g., A-B) and returns it as a dictionary. """
def read_letter_pair_scores(target_letter, compare_letter):
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

def read_word_averages():
    filename = os.path.join("CSV Files", "word_averages.csv")

    if not os.path.exists(filename):
        logging.warning(f"Word averages file '{filename}' does not exist.")
        return None

    word_averages = defaultdict(float)
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading Word Averages", unit=" word", colour="blue"):
            word_averages[row["Word"]] = float(row["Avg_Score"])

    return word_averages

def lookup_word_pair_score(word1, word2, csv_dir="CSV Files"):
    """
    Efficiently look up the score for a word pair using the CSV file for their starting letters.
    ####Optimization: If the 'Source' doesn't match, skip N rows where N = number of words starting with the 'Target' letter.
    """
    l1, l2 = word1[0].upper(), word2[0].upper()
    if l1 == l2:
        logging.warning(f"Both words start with the same letter: {l1}. Cannot look up score.")
        return None

    # Always sort letters to match filename convention
    sorted_letters = sorted([l1, l2])
    filename = os.path.join(csv_dir, WORD_PAIR_FILENAME_TEMPLATE.format(*sorted_letters))
    if not os.path.exists(filename):
        logging.warning(f"Distance matrix file '{filename}' does not exist.")
        return None

    # Determine which word is Source and which is Target in the file
    if l1 < l2:
        source_word, target_word = word1, word2
    else:
        source_word, target_word = word2, word1

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader, desc=f"Looking up {source_word}-{target_word} score", unit="row", colour="blue"):
            # Skip rows until we find the source word or run out of rows
            if (row["Source"] == source_word and row["Target"] == target_word) or (row["Source"] == target_word and row["Target"] == source_word):
                return {
                    "levenshtein": float(row["Levenshtein Distance"]),
                    "phoneme": float(row["Phoneme Distance"]),
                    "shared": int(row["Shared Sequence Count"]),
                    "score": float(row["Score"])
                    }

    logging.warning(f"No score found for {word1}-{word2}.")
    return None

def read_phoneme_difference_file(filename):
    if os.path.exists(filename):
        logging.info(f"Loading phoneme distances from '{filename}'...")
        distances = defaultdict(float)
        with open(filename, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                p1, p2, dist = row
                distances[(p1, p2)] = distances[(p2, p1)] = float(dist)
        return distances
    else:
        logging.error(f"Phoneme distance file '{filename}' does not exist!")
        return None

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
    min_phoneme_length = settings["min_phoneme_length"]
    max_phoneme_length = settings["max_phoneme_length"]
    min_word_length = settings["min_word_length"]
    max_word_length = settings["max_word_length"]
    min_syllables = settings["min_syllables"]
    max_syllables = settings["max_syllables"]

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
                (w not in WORD_BLACKLIST) and
                (w.isalpha()) and
                (min_phoneme_length <= len(prons[0]) <= max_phoneme_length) and
                (min_word_length <= len(w) <= max_word_length) and
                (min_syllables <= len([p for p in prons[0] if p[-1].isdigit()]) <= max_syllables) and
                (wordnet.synsets(w)) and
                (not any(w.startswith(prefix) for prefix in PREFIX_FILTERS)) and
                (not any(w.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(w[0].lower(), []))) and
                (word_averages[w] >= min_avg if word_averages else True)
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

def main():
    log_console_header("Loading and Cleaning CMU Dictionary")
    # Load the CMU Pronouncing Dictionary and create the phoneme distance dictionary
    nltk.download('cmudict')
    nltk.download('wordnet')

    PHONEME_DICT = get_cleaned_cmu_dict()
    logging.info("Phoneme Dictionary Loaded with %d words", len(PHONEME_DICT))
    PHONEME_DICT_NORMALIZED = {w: normalize_phoneme(pron) for w, pron in PHONEME_DICT.items()}
    logging.info("Phoneme Dictionary Normalized")

    logging.info("Total Words: {:,}".format(len(PHONEME_DICT)))
    
    PHONEME_DISTANCE_DICT = get_phoneme_coord_distance_dict()
    PHONEME_AUDIO_DISTANCE_DICT = get_phoneme_audio_difference_dict()
    PHONEME_COORD_MATRIX, PHONEME_MATRIX_INDEX = build_phoneme_distance_matrix(PHONEME_DISTANCE_DICT)
    PHONEME_AUDIO_MATRIX, _ = build_phoneme_distance_matrix(PHONEME_AUDIO_DISTANCE_DICT)

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
        write_word_pair_scores(PHONEME_DICT, WORDS_BY_LETTER)

    elif choice == "Generate Word Averages":
        write_word_averages(PHONEME_DICT, WORDS_BY_LETTER)

    elif choice == "Find Best (Randomized Trial)":
        log_console_header("Finding Best Phonetic Alphabet via Randomized Trial...")

        TRIALS = 1000000 # load_user_settings()["trials"]
        best_scores = find_best_set_randomized(PHONEME_DICT, WORDS_BY_LETTER, TRIALS, top_candidates=TRIALS)
        log_console_header(f"Best Scores Found in {TRIALS} Trials")
        logging.info("Average Calculation Time: %.6f seconds", best_scores['avg_calc_time'])
        
        # Log best sets
        log_scores("Overall", best_scores['score'], PHONEME_DICT)
    
    elif choice == "Score Premade Alphabet":
        NATO = [w for w in NATO_PHONETIC_ALPHABET if w in PHONEME_DICT]
        best_scores = _score_candidate(NATO, PHONEME_DICT)
        logging.info("--------------------------------")
        logging.info(f"NATO Score ({best_scores['score']:,.2f}):")
        logging.info("--------------------------------")
        for word in NATO:
            logging.info("%-8s  ->  %s", word.capitalize(), ' '.join(PHONEME_DICT[word][0]))

    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()