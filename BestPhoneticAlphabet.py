import os
import csv
import math
import random
import logging
import json
import string
from collections import defaultdict
import concurrent.futures

# Scientific and Data Libraries
import numpy as np
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

# Distance and Edit Libraries
import editdistance
from fastdtw import fastdtw

# Audio Processing
import librosa

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
LETTER_PAIR_FILENAME:str = "letter_pair_averages.csv"
CSV_DISTANCE_HEADERS:list[str] = ["Source", "Target", "Score",
        "Source-Phonemes", "Target-Phonemes",
        "Levenshtein Distance", "Phoneme Distance", 
        "Shared Sequence Count", "Sequences",
        "Suffix Count", "Suffixes",
        "Rhyme Count", "Rhymes",
        "Phoneme Audio Distance"]
CSV_DISTANCE_AVERAGE_HEADERS:list[str] = ["Source", "Target", "Total Word Pairs",
                            "Min Lev", "Max Lev", "Avg Lev", "Std Dev Lev",
                            "Min Phoneme", "Max Phoneme", "Avg Phoneme", "Std Dev Phoneme",
                            "Min Shared", "Max Shared", "Avg Shared", "Std Dev Shared",
                            "Min Score", "Max Score", "Avg Score", "Std Dev Score"]

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
PHONEME_DISTANCE_DICT = defaultdict()
PHONEME_AUDIO_DISTANCE_DICT = defaultdict()
WORDS_BY_LETTER = defaultdict(list)

# --------------------------------
# Calculation Functions
# TODO: Extract these to separate file/module probably maybw
# --------------------------------

"""_summary_
Extract the rhyme portion of a phoneme list.
NOTE: Requires the phoneme list to have stress markers (e.g., '1' for primary stress).
Probably doesn't really work, doh well
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
Normalize a phoneme list by removing stress digits.
"""
def normalize_phoneme(p_list):
    return [p[:-1] if p[-1].isdigit() else p for p in p_list]

def candidate_gen(trials, letters, words_by_letter, preselected_by_letter=None):
    for _ in range(trials):
        candidate = []
        for le in letters:
            if le in preselected_by_letter:
                candidate.append(preselected_by_letter[le])
            elif words_by_letter[le]:
                candidate.append(random.choice(words_by_letter[le]))
        if len(candidate) == len(letters):
            yield candidate

def _score_candidate_unpack(args):
    return _score_candidate(*args)

def _score_candidate(selected_words, p_dict, p_distance_dict, p_audio_dist_dict, phoneme_suffix_length=2, weights=None):
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
    WEIGHT_LEVENSHTEIN =            weights["weight_levenshtein"] if weights is not None           else 1.0
    WEIGHT_PHONEME =                weights["weight_phoneme"] if weights is not None               else 1.0
    WEIGHT_SHARED_SEQ =             weights["weight_shared_seq"] if weights is not None            else 2.0
    WEIGHT_SHARED_SUFFIX =          weights["weight_shared_suffix"] if weights is not None         else 2.0
    WEIGHT_RHYME =                  weights["weight_rhyme"] if weights is not None                 else 2.0
    WEIGHT_VOWEL_DIVERSITY =        weights["weight_vowel_diversity"] if weights is not None       else 1.0
    WEIGHT_CONSONANT_DIVERSITY =    weights["weight_consonant_diversity"] if weights is not None   else 1.0
    WEIGHT_AUDIO_DIVERSITY =        weights["weight_audio_diversity"] if weights is not None       else 1.0

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
        "vowel_diversity": vowel_norm,
        "phoneme_audio_distance": audio_norm
    }

# --------------
# Main Functions
# TODO: Extract these to a separate file/module
# --------------

"""_summary_
Find the best set of words via random sampling.
Randomized search for best phonetic alphabet, allowing for preselected words.
preselected_words: list of words to lock in for their starting letter (case-insensitive).
"""
def find_best_set_randomized(p_dict, p_distance_dict, p_audio_dist_dict, words_by_letter, trials=1000, batch_size=1000, preselected_words=None):
    
    log_console_header("Starting Randomized Search for Best Set of Words", trials)
    
    # Track top 100 candidates for each metric
    TOP_N = 100
    top_candidates = {
        "score": [],           # Max (higher is better)
        "levenshtein": [],     # Max (higher is better)
        "phoneme": [],         # Max (higher is better)
        "shared_sequence": [], # Min (lower is better)
        "shared_suffix": []    # Min (lower is better)
    }

    letters = list(string.ascii_uppercase)

    # Handle preselected words
    preselected_by_letter = {}
    if preselected_words:
        for w in preselected_words:
            le = w[0].upper()
            preselected_by_letter[le] = min(preselected_by_letter[le], w) if le in preselected_by_letter else w
    batch = []
    candidates_processed = 0
    
    for c in tqdm(candidate_gen(trials, letters, words_by_letter, preselected_by_letter), 
                  total=trials, desc="Generating and Scoring Candidates", unit=" candidate", colour="green"):

        batch.append(c)
        if len(batch) >= batch_size:
            # Reduce memory footprint for each thread/worker 
            # (probably doesn't actually do anything, but maybe? Feels like something that'd be useful to speed things up)

            # Build a minimal p_dict for this batch by using only the words in the current batch.
            # Filter the Words' Phoneme List Dictionary (p_dict) to only include the phoneme lists of the words in the current batch
            all_candidate_words = set(w for candidate in batch for w in candidate)
            mini_p_dict = {w: p_dict[w] for w in all_candidate_words}

            # Grab all unique phonemes used in the current batch
            all_phonemes = set()
            for w in mini_p_dict:
                all_phonemes.update(p for p in mini_p_dict[w][0])

            # Create minimal distance dictionaries for the current batch
            mini_p_distance_dict = {
                (ph1, ph2): p_distance_dict[(ph1, ph2)]
                for ph1 in all_phonemes for ph2 in all_phonemes if (ph1, ph2) in p_distance_dict
            }
            mini_p_audio_dist_dict = {
                (ph1, ph2): p_audio_dist_dict[(ph1, ph2)]
                for ph1 in all_phonemes for ph2 in all_phonemes if (ph1, ph2) in p_audio_dist_dict
            }

            # Prepare arguments for each candidate
            args = [(candidate, mini_p_dict, mini_p_distance_dict, mini_p_audio_dist_dict) for candidate in batch]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(_score_candidate_unpack, args),
                    total=len(args), desc=f"Scoring Candidates in Batch | Processed: {candidates_processed}",
                    unit=" candidate", leave=False, colour="blue"
                ))
                
            # Update top candidates for each metric
            for c, data in zip(batch, results):
                candidate_entry = {
                    "candidate": sorted(list(c)),
                    "score": data["score"],
                    "total_levenshtein": data["total_levenshtein"],
                    "total_phoneme_distance": data["total_phoneme_distance"],
                    "shared_sequence": data["shared_sequence"][0],
                    "shared_suffix": data["shared_suffix"][0]
                }
                
                # Define metrics with their data keys and sort directions
                metrics_config = {
                    "score": (data["score"], True),  # (value, reverse_sort)
                    "levenshtein": (data["total_levenshtein"], True),
                    "phoneme": (data["total_phoneme_distance"], True),
                    "shared_sequence": (data["shared_sequence"][0], False),
                    "shared_suffix": (data["shared_suffix"][0], False)
                }
                
                for metric_name, (value, reverse_sort) in metrics_config.items():
                    top_candidates[metric_name].append((value, candidate_entry))
                    top_candidates[metric_name].sort(key=lambda x: x[0], reverse=reverse_sort)
                    if len(top_candidates[metric_name]) > TOP_N:
                        top_candidates[metric_name].pop()

            candidates_processed += len(batch)
            batch = []

    # Write all top candidates to CSV
    log_console_header("Writing Top Candidates to CSV")
    headers = ["Metric", "Rank", "Value", "Score", "Total Levenshtein Distance", "Total Phoneme Distance", 
               "Total Shared Sequences", "Total Shared Suffixes"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    with open("best_random_search.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write top candidates for each metric
        for metric_name, candidates in top_candidates.items():
            for rank, (value, entry) in enumerate(candidates, 1):
                row = [
                    metric_name,
                    rank,
                    value,
                    entry["score"],
                    entry["total_levenshtein"],
                    entry["total_phoneme_distance"],
                    entry["shared_sequence"],
                    entry["shared_suffix"]
                ] + entry["candidate"]
                writer.writerow(row)

    # Return best scores in original format for compatibility
    best_scores = {
        "levenshtein": (top_candidates["levenshtein"][0][0], top_candidates["levenshtein"][0][1]["candidate"]) if top_candidates["levenshtein"] else (float('-inf'), []),
        "phoneme": (top_candidates["phoneme"][0][0], top_candidates["phoneme"][0][1]["candidate"]) if top_candidates["phoneme"] else (float('-inf'), []),
        "seq": (top_candidates["shared_sequence"][0][0], top_candidates["shared_sequence"][0][1]["candidate"]) if top_candidates["shared_sequence"] else (float('inf'), []),
        "suffix": (top_candidates["shared_suffix"][0][0], top_candidates["shared_suffix"][0][1]["candidate"]) if top_candidates["shared_suffix"] else (float('inf'), []),
        "score": (top_candidates["score"][0][0], top_candidates["score"][0][1]["candidate"]) if top_candidates["score"] else (float('-inf'), []),
    }

    log_console_header(f"Top {TOP_N} candidates for each metric saved to 'best_random_search.csv'")
    return best_scores

def genetic_algorithm_phonetic_alphabet(
    p_dict, 
    p_distance_dict, 
    p_audio_dist_dict, 
    words_by_letter, 
    population_size=100,
    generations=1000,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elite_size=10,
    tournament_size=5
):
    """
    Genetic Algorithm to find the best 26-word phonetic alphabet.
    
    Args:
        p_dict: Phoneme dictionary
        p_distance_dict: Phoneme distance dictionary  
        p_audio_dist_dict: Audio distance dictionary
        words_by_letter: Dictionary of words grouped by first letter
        population_size: Number of individuals in population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
        elite_size: Number of best individuals to keep each generation
        tournament_size: Size of tournament for selection
    
    Returns:
        Best individual found and its fitness score
    """
    log_console_header("Starting Genetic Algorithm for Phonetic Alphabet")
    
    letters = list(string.ascii_uppercase)
    
    def create_individual():
        """Create a random individual (26-word alphabet)"""
        return [random.choice(words_by_letter[letter]) for letter in letters]
    
    def fitness(individual):
        """Calculate fitness score for an individual"""
        try:
            result = _score_candidate(individual, p_dict, p_distance_dict, p_audio_dist_dict)
            return result["score"]
        except Exception:
            return float('-inf')  # Invalid individual
    
    def tournament_selection(population, fitness_scores, tournament_size):
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]
    
    def crossover(parent1, parent2):
        """Single-point crossover between two parents"""
        if random.random() > crossover_rate:
            return parent1[:], parent2[:]  # No crossover
        
        crossover_point = random.randint(1, 25)  # Don't include endpoints
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def mutate(individual):
        """Mutate an individual by changing some words"""
        mutated = individual[:]
        for i in range(26):
            if random.random() < mutation_rate:
                letter = letters[i]
                mutated[i] = random.choice(words_by_letter[letter])
        return mutated
    
    # Initialize population
    log_console_header("Initializing Population", population_size)
    population = [create_individual() for _ in range(population_size)]
    
    # Track best individuals across all generations
    best_ever_individual = None
    best_ever_fitness = float('-inf')
    generation_stats = []
    
    # Evolution loop
    for generation in tqdm(range(generations), desc="Evolving Generations", unit="generation", colour="green", leave=False):
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in tqdm(population, desc=f"Gen {generation+1}: Evaluating Fitness", unit="individual", colour="blue", leave=False):
            fitness_scores.append(fitness(individual))
        
        # Track statistics
        max_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        min_fitness = np.min(fitness_scores)
        
        generation_stats.append({
            'generation': generation,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'min_fitness': min_fitness
        })
        
        # Update best ever
        if max_fitness > best_ever_fitness:
            best_ever_fitness = max_fitness
            best_ever_individual = population[fitness_scores.index(max_fitness)][:]
        
        # Selection and reproduction
        new_population = []
        
        # Elitism: Keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx][:])
        
        # Generate offspring to fill rest of population
        while len(new_population) < population_size:
            # Select parents
            parent1 = tournament_selection(population, fitness_scores, tournament_size)
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Final evaluation and results
    log_console_header("Genetic Algorithm Complete")
    log_console_header("Best Individual Found", f"Fitness: {best_ever_fitness:.2f}")
    
    # Save results to CSV
    with open("genetic_algorithm_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write generation statistics
        writer.writerow(["Generation", "Max Fitness", "Avg Fitness", "Min Fitness"])
        for stats in generation_stats:
            writer.writerow([stats['generation'], stats['max_fitness'], 
                           stats['avg_fitness'], stats['min_fitness']])
        
        writer.writerow([])  # Empty row
        writer.writerow(["Best Individual Found:"])
        writer.writerow(["Letter", "Word", "Phonemes"])
        
        for i, word in enumerate(best_ever_individual):
            letter = letters[i]
            phonemes = ' '.join(p_dict[word][0]) if word in p_dict else "N/A"
            writer.writerow([letter, word, phonemes])
    
    log_console_header("Results saved to 'genetic_algorithm_results.csv'")
    
    return best_ever_individual, best_ever_fitness

def genetic_algorithm_with_preselected(
    p_dict, 
    p_distance_dict, 
    p_audio_dist_dict, 
    words_by_letter,
    preselected_words=None,
    **kwargs
):
    """
    Genetic Algorithm variant that respects preselected words.
    
    Args:
        preselected_words: List of words that must be included (locks their letters)
        **kwargs: Other parameters passed to main GA function
    """
    
    # Process preselected words
    preselected_by_letter = {}
    if preselected_words:
        for word in preselected_words:
            letter = word[0].upper()
            preselected_by_letter[letter] = word.lower()
    
    letters = list(string.ascii_uppercase)
    
    def create_individual():
        """Create individual respecting preselected constraints"""
        individual = []
        for letter in letters:
            if letter in preselected_by_letter:
                individual.append(preselected_by_letter[letter])
            else:
                individual.append(random.choice(words_by_letter[letter]))
        return individual
    
    def mutate(individual):
        """Mutate only non-preselected positions"""
        mutated = individual[:]
        mutation_rate = kwargs.get('mutation_rate', 0.1)
        
        for i in range(26):
            letter = letters[i]
            if letter not in preselected_by_letter and random.random() < mutation_rate:
                mutated[i] = random.choice(words_by_letter[letter])
        return mutated
    
    # Override functions in kwargs
    kwargs['create_individual'] = create_individual
    kwargs['mutate'] = mutate
    
    log_console_header("Starting Genetic Algorithm with Preselected Words", 
                      f"Locked letters: {list(preselected_by_letter.keys())}")
    
    return genetic_algorithm_phonetic_alphabet(
        p_dict, p_distance_dict, p_audio_dist_dict, words_by_letter, **kwargs
    )

# --------------
# WRITE CSV FUNCTIONS
# TODO: Extract these to a separate file/module
# --------------

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
                                        data["rhyme"][0], data["rhyme"][1],
                                        data["phoneme_audio_distance"]])   
                    
                        gephi_writer.writerow([w1, w2, data["score"]])  # Write the data to the Gephi CSV file

"""_summary_
Write the average distances for each letter pair to a CSV file.
This function reads all the CSV files in the "CSV Files" directory and calculates the averages for each letter pair.
The results are written to a new CSV file named "letter_pair_averages.csv".
"""
def write_letter_pair_averages_csv():
    
    letters = list(string.ascii_uppercase)
    avg_filename = os.path.join("CSV Files", LETTER_PAIR_FILENAME)
    
    with open(avg_filename, "w", newline="") as avg_file:
        avg_writer = csv.writer(avg_file)
        avg_writer.writerow(CSV_DISTANCE_AVERAGE_HEADERS)

    for l1 in tqdm(letters, total=len(letters), desc="Calculating Letter Pair Averages", unit=" letter", colour="green"):
        for l2 in tqdm(letters, total=len(letters), desc=f"Calculating {l1}-Letter Pair Averages", unit=" letter pair", leave=False, colour="yellow"):
            
            if l1 == l2:
                continue
      
            data = {
                "levenshtein": [],
                "phoneme": [],
                "shared": [],
                "score": []
            }
            
            letter1, letter2 = sorted([l1, l2])  # Sort to ensure consistent order
            word_pairs = read_distance_matrix(letter1, letter2)
            for w1 in tqdm(word_pairs, total=len(word_pairs), desc=f"Processing {letter1}-{letter2} Word Pairs", unit=" word", leave=False, colour="red"):
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

def write_letter_averages_csv():
    """
    Write the average distances for each letter to a CSV file.
    This function reads all the CSV files in the "CSV Files" directory and calculates the averages for each letter.
    The results are written to a new CSV file named "letter_averages.csv".
    """
    letters = list(string.ascii_uppercase)
    avg_filename = os.path.join("CSV Files", "letter_averages.csv")

    with open(avg_filename, "w", newline="") as avg_file:
        avg_writer = csv.writer(avg_file)
        avg_writer.writerow(["Letter", 
                            "Count",
                            "Levenshtein Min", "Levenshtein Max", "Levenshtein Avg", "Levenshtein Std",
                            "Phoneme Min", "Phoneme Max", "Phoneme Avg", "Phoneme Std",
                            "Shared Min", "Shared Max", "Shared Avg", "Shared Std",
                            "Score Min", "Score Max", "Score Avg", "Score Std"])

    for l1 in tqdm(letters, total=len(letters), desc="Calculating Letter Averages", unit=" letter", colour="green"):
        data = {
            "levenshtein": [],
            "phoneme": [],
            "shared": [],
            "score": []
        }

        for l2 in tqdm(letters, total=len(letters), desc=f"Calculating {l1}-Letter Averages", unit=" letter pair", leave=False, colour="yellow"):
            if l1 == l2:
                continue
            
            # Read the distance matrix for the letter pair (letter, l2)
            # This will read the CSV file for the letter pair
            letter1, letter2 = sorted([l1, l2])  # Sort to ensure consistent order
            word_pairs = read_distance_matrix(letter1, letter2)
            for w1 in tqdm(word_pairs, total=len(word_pairs), desc=f"Processing {l1} Word Pairs", unit=" word", leave=False, colour="red"):
                for w2 in word_pairs[w1]:
                    data["levenshtein"].append(word_pairs[w1][w2]["levenshtein"])
                    data["phoneme"].append(word_pairs[w1][w2]["phoneme"])
                    data["shared"].append(word_pairs[w1][w2]["shared"])
                    data["score"].append(word_pairs[w1][w2]["score"])

        # Write the Letter Averages and other data points to the Letter Averages CSV file
        with open(avg_filename, "a", newline="") as avg_file:
            avg_writer = csv.writer(avg_file)
            l_min, l_max, l_avg, l_std = np.min(data["levenshtein"]), np.max(data["levenshtein"]), np.mean(data["levenshtein"]), np.std(data["levenshtein"])
            p_min, p_max, p_avg, p_std = np.min(data["phoneme"]), np.max(data["phoneme"]), np.mean(data["phoneme"]), np.std(data["phoneme"])
            s_min, s_max, s_avg, s_std = np.min(data["shared"]), np.max(data["shared"]), np.mean(data["shared"]), np.std(data["shared"])
            x_min, x_max, x_avg, x_std = np.min(data["score"]), np.max(data["score"]), np.mean(data["score"]), np.std(data["score"])
            avg_writer.writerow([l1, len(data["levenshtein"]),
                                l_min, l_max, l_avg, l_std,
                                p_min, p_max, p_avg, p_std,
                                s_min, s_max, s_avg, s_std,
                                x_min, x_max, x_avg, x_std])

def write_word_averages_csv(p_dict, p_distance_dict, p_audio_dist_dict, words_by_letter):
    
    log_console_header("Calculating Word Averages")
    csv_filename = os.path.join("CSV Files", "word_averages.csv")

    # Create the CSV file and write the header. We'll write the results as we calculate
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)    
        # Write header. 
        # TODO: Extract this into it's own variable like the others
        headers = [
            "Word", "Letter", "Phonemes", "Phoneme_Count", "Phoneme_Magnitude", "Comparisons_Made",
            "Avg_Score", "Min_Score", "Max_Score", "Std_Score",
            "Avg_Levenshtein", "Min_Levenshtein", "Max_Levenshtein", "Std_Levenshtein",
            "Avg_Phoneme_Distance", "Min_Phoneme_Distance", "Max_Phoneme_Distance", "Std_Phoneme_Distance",
            "Avg_Shared_Sequences", "Min_Shared_Sequences", "Max_Shared_Sequences", "Std_Shared_Sequences",
            "Avg_Shared_Suffixes", "Min_Shared_Suffixes", "Max_Shared_Suffixes", "Std_Shared_Suffixes",
            "Avg_Rhymes", "Min_Rhymes", "Max_Rhymes", "Std_Rhymes"
        ]
        writer.writerow(headers)

    # For loops written in a way so we don't constantly go
    #   "gimmie the dictionary with all the words that don't start with this letter" for each word
    # probably a better way to do this, bleh
    for letter1 in tqdm(string.ascii_uppercase, desc="Processing Letters", unit="letter", colour="green"):
        word_averages = []
       
        for word in tqdm(words_by_letter[letter1], desc="Calculating word averages", unit="word", colour="green"):     
            scores = []
            levenshtein_scores = []
            phoneme_scores = []
            shared_sequence_scores = []
            shared_suffix_scores = []
            rhyme_scores = []
            
            # Calculate phoneme magnitude for the word
            phonemes = normalize_phoneme(p_dict[word][0])
            phoneme_magnitude = 0.0
            for p in phonemes:
                if p in PHONEME_COORDINATES:
                    coords = PHONEME_COORDINATES[p]
                    magnitude = np.sqrt(sum(coord**2 for coord in coords))
                    phoneme_magnitude += magnitude

            # Calculate scores against all other words starting with different letters
            word_letter = word[0].upper()    
            for letter2 in tqdm(string.ascii_uppercase, desc=f"Comparing {word_letter} with other letters", unit="letter", leave=False, colour="red"):
                if letter2 == letter1:
                    continue
                for other_word in tqdm(words_by_letter[letter2], desc=f"Scoring {word}", unit="comparison", leave=False, colour="blue"):
                    result = _score_candidate([word, other_word], p_dict, p_distance_dict, p_audio_dist_dict)
                    scores.append(result["score"])
                    levenshtein_scores.append(result["total_levenshtein"])
                    phoneme_scores.append(result["total_phoneme_distance"])
                    shared_sequence_scores.append(result["shared_sequence"][0])
                    shared_suffix_scores.append(result["shared_suffix"][0])
                    rhyme_scores.append(result["rhyme"][0])
            
            word_averages.append({
                'word': word,
                'letter': word_letter,
                'phonemes': ' '.join(p_dict[word][0]),
                'phoneme_count': len(p_dict[word][0]),
                'phoneme_magnitude': phoneme_magnitude,
                'comparisons_made': len(scores),
                'avg_score': np.mean(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'std_score': np.std(scores),
                'avg_levenshtein': np.mean(levenshtein_scores),
                'min_levenshtein': np.min(levenshtein_scores),
                'max_levenshtein': np.max(levenshtein_scores),
                'std_levenshtein': np.std(levenshtein_scores),
                'avg_phoneme_distance': np.mean(phoneme_scores),
                'min_phoneme_distance': np.min(phoneme_scores),
                'max_phoneme_distance': np.max(phoneme_scores),
                'std_phoneme_distance': np.std(phoneme_scores),
                'avg_shared_sequences': np.mean(shared_sequence_scores),
                'min_shared_sequences': np.min(shared_sequence_scores),
                'max_shared_sequences': np.max(shared_sequence_scores),
                'std_shared_sequences': np.std(shared_sequence_scores),
                'avg_shared_suffixes': np.mean(shared_suffix_scores),
                'min_shared_suffixes': np.min(shared_suffix_scores),
                'max_shared_suffixes': np.max(shared_suffix_scores),
                'std_shared_suffixes': np.std(shared_suffix_scores),
                'avg_rhymes': np.mean(rhyme_scores),
                'min_rhymes': np.min(rhyme_scores),
                'max_rhymes': np.max(rhyme_scores),
                'std_rhymes': np.std(rhyme_scores)
            })

        # Append the current word's metrics to the CSV file
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            # Write data rows
            for word_data in tqdm(word_averages, desc="Writing word data", unit="word", colour="yellow"):
                row = [
                    word_data['word'],
                    word_data['letter'],
                    word_data['phonemes'],
                    word_data['phoneme_count'],
                    word_data['phoneme_magnitude'],
                    word_data['comparisons_made'],
                    word_data['avg_score'],
                    word_data['min_score'],
                    word_data['max_score'],
                    word_data['std_score'],
                    word_data['avg_levenshtein'],
                    word_data['min_levenshtein'],
                    word_data['max_levenshtein'],
                    word_data['std_levenshtein'],
                    word_data['avg_phoneme_distance'],
                    word_data['min_phoneme_distance'],
                    word_data['max_phoneme_distance'],
                    word_data['std_phoneme_distance'],
                    word_data['avg_shared_sequences'],
                    word_data['min_shared_sequences'],
                    word_data['max_shared_sequences'],
                    word_data['std_shared_sequences'],
                    word_data['avg_shared_suffixes'],
                    word_data['min_shared_suffixes'],
                    word_data['max_shared_suffixes'],
                    word_data['std_shared_suffixes'],
                    word_data['avg_rhymes'],
                    word_data['min_rhymes'],
                    word_data['max_rhymes'],
                    word_data['std_rhymes']
                ]
                writer.writerow(row)

    
    log_console_header(f"Word averages saved to '{csv_filename}'")
    return word_averages

# -------------------------------
# READ CSV Functions
# TODO: Extract Read file functions into a separate module
# -------------------------------

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

def read_word_averages_csv():
    filename = os.path.join("CSV Files", "word_averages.csv")
    word_averages = defaultdict(float)

    if not os.path.exists(filename):
        logging.warning(f"Word averages file '{filename}' does not exist.")
        return word_averages

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading Word Averages", unit="word", colour="blue"):
            word_averages[row["Word"]] =  float(row["Avg_Score"])

    return word_averages

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

    # Get user settings
    settings = load_user_settings()
    min_phoneme_length = settings["min_phoneme_length"]
    max_phoneme_length = settings["max_phoneme_length"]
    min_word_length = settings["min_word_length"]
    max_word_length = settings["max_word_length"]
    min_syllables = settings["min_syllables"]
    max_syllables = settings["max_syllables"]

    #word_averages = read_word_averages_csv()

    cleaned_dict = cmudict.dict()
    cleaned_dict = {
        word: prons for word, prons in tqdm(cleaned_dict.items(), desc="Cleaning CMU Dictionary", unit="word")
        if (
            (word in WORD_WHITELIST) or
            (word.isalpha()) and
            (word not in WORD_BLACKLIST) and
            (not any(word.startswith(prefix) for prefix in PREFIX_FILTERS)) and
            (len(prons[0]) >= min_phoneme_length) and
            (len(prons[0]) <= max_phoneme_length) and
            (len(word) >= min_word_length) and
            (len(word) <= max_word_length) and
            (len(set(word)) > 1) and
            (not any(word.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(word[0].lower(), []))) and
            (wordnet.synsets(word)) and
            (min_syllables <= len([p for p in prons[0] if p[-1].isdigit()]) <= max_syllables)
            #(word_averages[word] >= 800.0)
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

"""_summary_
Log the best scores and words in a particular format.
"""
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

    user_settings = load_user_settings()
    TRIALS = user_settings["trials"]

    log_console_header("Loading and Cleaning CMU Dictionary")
    # Load the CMU Pronouncing Dictionary and create the phoneme distance dictionary
    nltk.download('cmudict')
    nltk.download('wordnet')

    PHONEME_DICT = get_cleaned_cmu_dict()
    PHONEME_DISTANCE_DICT = get_phoneme_distance_dict()

    PHONEME_AUDIO_DISTANCE_DICT = get_phoneme_audio_difference_dict("Phoneme Voice Files")
    WORDS_BY_LETTER, WORDS_BY_LENGTH, WORDS_BY_PHONEME_LENGTH, WORDS_BY_SYLLABLE = get_words_data(PHONEME_DICT)

    log_console_header("Dictionary Stats: ")
    logging.info("Total Words: {:,}".format(len(PHONEME_DICT)))
    for l in LETTERS:
        logging.info("'%-3s': %d words", l, len(WORDS_BY_LETTER[l]))

    # Calculate how many edges there would be if we connected every word to every other word minus same-letter words
    total_edges = 0

    # Count edges between each pair of letters
    for i, letter1 in enumerate(LETTERS):
        for j, letter2 in enumerate(LETTERS):
            if i < j:  # Avoid double counting (A-B and B-A are the same edge)
                words1 = len(WORDS_BY_LETTER[letter1])
                words2 = len(WORDS_BY_LETTER[letter2])
                total_edges += words1 * words2
    logging.info("Total Edges (excluding same-letter connections): {:,}".format(total_edges))

    # Calculate how many possible phonetic alphabets we could construct
    total_candidates = 1
    for letter in LETTERS:
        total_candidates *= len(WORDS_BY_LETTER[letter])

    logging.info("Total Candidates for Phonetic Alphabet: {:.2e} ({:,})".format(total_candidates, total_candidates))

    choices = {
        '1': "Generate Distance Matrices",
        '2': "Generate Letter-Pair Averages",
        '3': "Generate Letter Averages",
        '4': "Generate Word Averages",
        '5': "Find Best (Randomized Trial)",
        '6': "Score Premade Alphabet",
        '7': "Genetic Algorithm Search",
        '8': "Genetic Algorithm with Constraints"
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
    
    elif choice == "Generate Letter-Pair Averages":
        write_letter_pair_averages_csv()
    
    elif choice == "Generate Letter Averages":
        write_letter_averages_csv()
    
    elif choice == "Generate Word Averages":
        write_word_averages_csv(PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT, WORDS_BY_LETTER)

    elif choice == "Find Best (Randomized Trial)":
        log_console_header("Finding Best Phonetic Alphabet via Randomized Trial")
        best_scores = find_best_set_randomized(PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT, WORDS_BY_LETTER, TRIALS)
        
        # Log best sets
        log_scores("Levenshtein Distance", best_scores['levenshtein'], PHONEME_DICT)
        log_scores("Phoneme Distance", best_scores['phoneme'], PHONEME_DICT)
        log_scores("Shared Phoneme Sequence Count", best_scores['seq'], PHONEME_DICT)
        log_scores("Overall", best_scores['score'], PHONEME_DICT)
    
    elif choice == "Score Premade Alphabet":
        NATO = [w for w in NATO_PHONETIC_ALPHABET if w in PHONEME_DICT]
        best_scores = _score_candidate(NATO, PHONEME_DICT, PHONEME_DISTANCE_DICT, PHONEME_AUDIO_DISTANCE_DICT)
        logging.info("--------------------------------")
        logging.info(f"NATO Score ({best_scores['score']:,.2f}):")
        logging.info("--------------------------------")
        for word in NATO:
            logging.info("%-8s  ->  %s", word.capitalize(), ' '.join(PHONEME_DICT[word][0]))
    
    elif choice == "Genetic Algorithm Search":
        log_console_header("Finding Best Phonetic Alphabet via Genetic Algorithm")
        
        # Basic genetic algorithm
        best_individual, best_fitness = genetic_algorithm_phonetic_alphabet(
            PHONEME_DICT, 
            PHONEME_DISTANCE_DICT, 
            PHONEME_AUDIO_DISTANCE_DICT, 
            WORDS_BY_LETTER,
            population_size=50,
            generations=500,
            mutation_rate=0.15,
            crossover_rate=0.8
        )
        
        # Log the results
        logging.info("--------------------------------")
        logging.info(f"Best GA Result (Fitness: {best_fitness:.2f}):")
        logging.info("--------------------------------")
        for i, word in enumerate(best_individual):
            letter = string.ascii_uppercase[i]
            phonemes = ' '.join(PHONEME_DICT[word][0])
            logging.info("%-3s: %-12s -> %s", letter, word.capitalize(), phonemes)
    
    # With preselected words
    elif choice == "Genetic Algorithm with Constraints":
        preselected = ["alpha", "bravo", "charlie"]  # Lock in some NATO words
        best_individual, best_fitness = genetic_algorithm_with_preselected(
            PHONEME_DICT, 
            PHONEME_DISTANCE_DICT, 
            PHONEME_AUDIO_DISTANCE_DICT, 
            WORDS_BY_LETTER,
            preselected_words=preselected,
            population_size=50,
            generations=300
        )

    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()