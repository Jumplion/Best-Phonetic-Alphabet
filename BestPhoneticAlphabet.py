import os
import csv
import math
import random
import logging
import itertools
from collections import defaultdict
import concurrent.futures

import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

import editdistance

# 🔧 CONFIGURATION
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detail, WARNING for less
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        # Uncomment below to also log to a file:
        # logging.FileHandler("phonetic_alphabet.log")
    ]
)

# -----------------------------
# Filter Options
# -----------------------------
FILTER_NON_ALPHABETIC =     True    # Set to 'True' to filter out words with non-alphabetic characters
FILTER_SINGLE_LETTER =      True    # Set to 'True' to filter out single letter words
FILTER_BY_MIN_PHONEME =     True    # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
FILTER_BY_MAX_PHONEME =     True    # Set to 'True' to filter out words with more than MAX_PHONEME_LENGTH phonemes
FILTER_BY_MIN_LENGTH =      True    # Set to 'True' to filter out words with fewer than MIN_WORD_LENGTH letters
FILTER_BY_MAX_LENGTH =      False   # Set to 'True' to filter out words with more than MAX_WORD_LENGTH letters
FILTER_BY_MIN_SYLLABLES =   True    # Set to 'True' to filter out words with fewer than MIN_SYLLABLES syllables
FILTER_BY_MAX_SYLLABLES =   True    # Set to 'True' to filter out words with more than MAX_SYLLABLES syllables

LEMMATIZE_DICT = True          # Set to 'True' to lemmatize words
LEMMATIZE_SEVERITY = 3         # 0 (None) | 1 (Noun) | 2 (Verb) | 3 (Aggressive) | Otherwise Aggressive
CHECK_AGAINST_WORDNET = True   # Set to 'True' to check if words exist in WordNet

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
# Randomization Options
# -----------------------------
TRIALS = 100000                  # Number of random trials
PENALIZE_LENGTH_VARIANCE = True # Set to 'True' to penalize length variance
LENGTH_VARIANCE_WEIGHT = 10     # Adjust this weight to control penalty severity

# -----------------------------
# Individual Prefix filters
# -----------------------------
FILTER_EU = True    # Set to 'True' to filter out words starting with "eu"
FILTER_GN = True    # Set to 'True' to filter out words starting with "gn"
FILTER_KN = True    # Set to 'True' to filter out words starting with "kn"
FILTER_MN = True    # Set to 'True' to filter out words starting with "mn"
FILTER_PH = True    # Set to 'True' to filter out words starting with "ph"
FILTER_PN = True    # Set to 'True' to filter out words starting with "pn"
FILTER_PS = True    # Set to 'True' to filter out words starting with "ps"
FILTER_SH = True    # Set to 'True' to filter out words starting with "sh"
FILTER_TH = True    # Set to 'True' to filter out words starting with "th"
FILTER_WR = True    # Set to 'True' to filter out words starting with "wr"

PREFIX_FILTERS = {
    "eu": FILTER_EU,
    "gn": FILTER_GN,
    "kn": FILTER_KN,
    "mn": FILTER_MN,
    "ph": FILTER_PH,
    "pn": FILTER_PN,
    "ps": FILTER_PS,
    "sh": FILTER_SH,
    "th": FILTER_TH,
    "wr": FILTER_WR
}

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
    "p" : [],
    "q" : [],
    "r" : [],
    "s" : [],
    "t" : [],
    "u" : [],
    "v" : [],    
    "w" : [],
    "x" : [],
    "y" : [],
    "z" : []
}

# Words that we don't want in our alphabets
# These are words that are either offensive, inappropriate, or just don't fit the criteria
# Some other reasons for blacklisting words:
# - Not caught by the filters
# - Strange pronunciations that don't fit the phonetic alphabet
# - Words that are too similar to other words in the alphabet
# - Slurs and racial remarks are a no-go, we diverse and tolerant in this sum-bitch
WORD_BLACKLIST = [
    "xhosa", 
    "xian", 
    "xinjiang", 
    "xenophobia", 
    "xenophobic", 
    "xenophon", 
    "iwo", 
    "pedophile", 
    "rapist",
]

WORD_WHITELIST = [
]

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
WORD_AVERAGES = defaultdict(dict)

# -----------------------------
# 📦 FUNCTION DEFINITIONS
# -----------------------------

# Calculate Levenshtein distance between two words.
def levenshtein_distance(w1, w2):
    return editdistance.eval(w1, w2)

# Calculate the phoneme distance between two phoneme sequences.
def phoneme_distance(p1_list, p2_list):
    distance = 0
    for i in range(min(len(p1_list), len(p2_list))):
        distance += PHONEME_DISTANCE_DICT.get((p1_list[i], p2_list[i]), 0)

    # Handle the case where one list is longer than the other
    remaining_phonemes = p1_list[len(p2_list):] if len(p1_list) > len(p2_list) else p2_list[len(p1_list):]
    # If the remaining phonemes are empty, return the distance
    if not remaining_phonemes:
        return distance
    # If the remaining phonemes are not empty, add the distance of the remaining phonemes
    # Adding the distance/magnitude of remaining phonemes are a flawed system, need to figure out a better way to handle this   
    distance += sum(PHONEME_DISTANCE_DICT.get((remaining_phonemes[i], ""), 0) for i in range(len(remaining_phonemes)))   
    return distance

# Calculate the number of shared phoneme sequences between two phoneme sequences.
def shared_phoneme_sequences(p1, p2):
    # Measure the time it takes to complete the function   
    set1, set2 = set(), set()
    n1, n2 = len(p1), len(p2)
    N = min(n1, n2) + 1

    for i in range(n1):
        for j in range(i + 1, min(n1, i + N)):
            set1.add(tuple(p1[i:j]))

    for i in range(n2):
        for j in range(i + 1, min(n2, i + N)):
            set2.add(tuple(p2[i:j]))

    # Count the intersection
    return len(set1 & set2)

# Calculate the total Levenshtein distance for a set of words (no parallel processing).
def total_levenshtein_distance(word_set):
    pairs = list(itertools.combinations(word_set, 2))
    distances = [levenshtein_distance(w1, w2) for w1, w2 in pairs]
    base_score = sum(distances)

    penalty = 0
    if PENALIZE_LENGTH_VARIANCE:
        lengths = [len(w) for w in word_set]
        std_dev = np.std(lengths)
        penalty = std_dev * LENGTH_VARIANCE_WEIGHT

    return base_score - penalty

# Calculate the total phoneme distance for a set of words.
# This function can calculate both basic phoneme distance and shared phoneme sequences.
def total_phoneme_distance(word_set, sequence=False):
    pairs = list(itertools.combinations(word_set, 2))
    func = phoneme_distance if not sequence else shared_phoneme_sequences
    distances = [func(PHONEME_DICT[w1][0], PHONEME_DICT[w2][0]) for w1, w2 in pairs]
    return sum(distances)

# Helper function for scoring a candidate (no parallel processing).
def _score_candidate(candidate):
    avg_levenshtein = total_levenshtein_distance(candidate) / len(candidate)
    avg_phoneme = total_phoneme_distance(candidate, False) / len(candidate)
    avg_shared = total_phoneme_distance(candidate, True) / len(candidate)
    score = avg_levenshtein + avg_phoneme - avg_shared
    return (score, avg_levenshtein, avg_phoneme, avg_shared, candidate)

# Find the best set of words via random sampling (no parallel processing).
def find_best_set_randomized(trials=1000): 
    logging.info("Starting Randomized Search for Best Set of Words...")
    best_score = best_score_levenshtein = best_score_phoneme = -1
    best_score_shared = 99999999999

    best_set = best_levenshtein = best_phoneme = best_shared = []

    letters = sorted(WORDS_BY_LETTER.keys())

    # Delete the file if it exists
    if os.path.exists("random_search_log.csv"):
        os.remove("random_search_log.csv")

    candidates = []
    for _ in range(trials):
        candidate = [random.choice(WORDS_BY_LETTER[letter]) for letter in letters if WORDS_BY_LETTER[letter]]
        if len(candidate) == 26:
            candidates.append(sorted(candidate))

    with open("random_search_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Score", "Avg Levenshtein Distance", "Avg Phoneme Distance (Basic)",
            "Avg Shared Phoneme Sequence Count"
        ] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

        # Sequential scoring of candidates
        for candidate in tqdm(candidates, total=len(candidates)):
            score, avg_levenshtein, avg_phoneme, avg_shared, candidate = _score_candidate(candidate)
            writer.writerow([score, avg_levenshtein, avg_phoneme, avg_shared] + candidate)

            if score > best_score:
                best_score = score
                best_set = candidate
            if avg_levenshtein > best_score_levenshtein:
                best_score_levenshtein = avg_levenshtein
                best_levenshtein = candidate
            if avg_phoneme > best_score_phoneme:
                best_score_phoneme = avg_phoneme
                best_phoneme = candidate
            if avg_shared < best_score_shared:
                best_score_shared = avg_shared
                best_shared = candidate

    best_scores = {
        "score": best_score,
        "best levenshtein": best_score_levenshtein,
        "best phoneme": best_score_phoneme,
        "best shared": best_score_shared
    }

    return best_scores, best_set, best_levenshtein, best_phoneme, best_shared

# Helper function for parallel processing of word distances.
def _word1_distances(args):
    word1, word1_pron, filtered_words, pron_dict, phoneme_distances = args
    total_dist_levenshtein = total_dist_phoneme = total_shared = count = 0.0
    
    for word2 in filtered_words:
        word2_pron = pron_dict[word2][0]
        total_dist_levenshtein += levenshtein_distance(word1, word2)
        total_dist_phoneme += sum(phoneme_distance(a, b) for a, b in itertools.zip_longest(word1_pron, word2_pron, fillvalue=""))
        total_shared += shared_phoneme_sequences(word1_pron, word2_pron)
        count += 1
    
    # Calculate the magnitude of the phoneme sequence
    magnitude = math.sqrt(sum(phoneme_distances.get((p1, p2), 0) ** 2 for p1, p2 in zip(word1_pron, word1_pron)))
    return magnitude, total_dist_levenshtein, total_dist_phoneme, total_shared, count

# Calculate and write word averages to a CSV file.
def write_word_averages(p_dict, p_distance_dict):
    if os.path.exists("word_score_averages.csv"):
        overwrite = input("File 'word_score_averages.csv' already exists. Recalculate and overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            logging.info("Exiting without overwriting the file.")
            return
        else:
            os.remove("word_score_averages.csv")
    
    log_console_header("Writing Word Averages to 'word_score_averages.csv'")
    with open("word_score_averages.csv", "w", newline="") as avg_csvfile:
        avg_writer = csv.writer(avg_csvfile)
        avg_writer.writerow(["Word", "Phonemes", "Magnitude", "Average Levenshtein Distance", "Average Phoneme Distance (Basic)", "Average Shared Phoneme Sequence Count"])
        word_list = list(p_dict.keys())

        log_console_header("Total Words to Process", len(word_list))
        args_list = []
        for word1 in tqdm(word_list, desc="Preparing Arguments for Parallel Processing", unit="word"):
            word1_pron = p_dict[word1][0]
            filtered_words = [w for w in word_list if w[0].upper() != word1[0].upper()]
            args_list.append((word1, word1_pron, filtered_words, p_dict, p_distance_dict))

        log_console_header("Starting Parallel Processing of Word Distances (this might take a minute)...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_word1_distances, args_list), total=len(args_list), desc="Calculating Word Scores", unit="word"))

        for ind, args in tqdm(enumerate(args_list), desc="Processing Results", total=len(args_list), unit="word"):
            word1, word1_pron = args[:2]
            magnitude, total_dist_levenshtein, total_dist_phoneme, total_shared, count = results[ind]
            avg_levenshtein = total_dist_levenshtein / count
            avg_phoneme = total_dist_phoneme / count
            avg_shared = total_shared / count
            word1_pron_str = " ".join(word1_pron)
            avg_writer.writerow([word1, word1_pron_str, magnitude, avg_levenshtein, avg_phoneme, avg_shared])
            logging.debug(
                "Word: %-12s | Phonemes: %-20s | Magnitude: %.6f | Avg Levenshtein: %.6f | Avg Phoneme Dist: %.6f | Avg Shared Seq Count: %.6f",
                word1, word1_pron_str, magnitude, avg_levenshtein, avg_phoneme, avg_shared
            )

        log_console_header("Word Averages Written to 'word_score_averages.csv'")

# Load word averages from a CSV file.
def load_word_averages(filename="word_score_averages.csv"):
    if not os.path.exists(filename):
        logging.error("File '%s' does not exist.", filename)
        return {}

    averages = {}
    with open(filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in tqdm(reader, desc="Loading Word Averages", unit="word"):
            word = row[0]
            phonemes = row[1]
            magnitude = float(row[2])
            avg_levenshtein = float(row[3])
            avg_phoneme = float(row[4])
            avg_shared = float(row[5])
            averages[word] = {"phonemes": phonemes, "magnitude": magnitude, "avg_levenshtein": avg_levenshtein, "avg_phoneme": avg_phoneme, "avg_shared": avg_shared }
    return averages

# Compare every phoneme coordinate with every other, as well as itself and an empty string
def get_phoneme_distance_dict():
    phonemes = list(PHONEME_COORDINATES.keys())
    distance = defaultdict(float)
    for i, p1 in enumerate(phonemes):
        coord1 = PHONEME_COORDINATES[p1]
        distance[(p1, "")] = distance[("", p1)] = math.sqrt(sum(a ** 2 for a in coord1))
        distance[(p1, p1)] = 0.0
        
        for j, p2 in enumerate(phonemes):
            if i < j:  # Avoid duplicate pairs
                coord2 = PHONEME_COORDINATES[p2]
                distance[(p1, p2)] = distance[(p2, p1)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                
    logging.info("Phoneme Distance Dictionary Created | Total Phonemes: %d", len(PHONEME_COORDINATES))
    # Print out the phoneme distance dictionary
    logging.debug("Phoneme Distance Dictionary: %s", distance)
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
            (FILTER_NON_ALPHABETIC and word.isalpha()) and
            (word not in WORD_BLACKLIST) and
            (not any(word.startswith(prefix) for prefix, active in PREFIX_FILTERS.items() if active)) and
            (not FILTER_BY_MIN_PHONEME or len(prons[0]) >= MIN_PHONEME_LENGTH) and
            (not FILTER_BY_MAX_PHONEME or len(prons[0]) <= MAX_PHONEME_LENGTH) and
            (not FILTER_BY_MIN_LENGTH or len(word) >= MIN_WORD_LENGTH) and
            (not FILTER_BY_MAX_LENGTH or len(word) <= MAX_WORD_LENGTH) and
            (not FILTER_SINGLE_LETTER or len(set(word)) > 1) and
            (not any(word.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(word[0].lower(), []))) and
            (not CHECK_AGAINST_WORDNET or wordnet.synsets(word)) and
            (
                not (FILTER_BY_MIN_SYLLABLES or FILTER_BY_MAX_SYLLABLES) or
                (MIN_SYLLABLES <= len([p for p in prons[0] if p[-1].isdigit()]) <= MAX_SYLLABLES)
            )
        )
    }              

    if LEMMATIZE_DICT:
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

    logging.info("CMU Pronouncing Dictionary Cleaned | Total Words: %d", len(cleaned_dict))

    return cleaned_dict

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------

def log_console_header(message, data=None):
    logging.info("--------------------------------")
    if data:
        logging.info("%s: %s", message, data)
    else:
        logging.info(message)
    logging.info("--------------------------------")

def main():

    log_console_header("Starting Best Phonetic Alphabet Search")

    # Load NLTK resources
    nltk.download('cmudict')
    nltk.download("wordnet")


    PHONEME_DICT = get_cleaned_cmu_dict()

# --------------------------------
# Print out Stats for Dictionary
# --------------------------------

    log_console_header("Total Words in Dictionary", len(PHONEME_DICT)) 
    
    WORDS_BY_LETTER = defaultdict(list)
    phoneme_count = defaultdict(int)
    letter_count = defaultdict(int)
    for word in PHONEME_DICT.keys():
        WORDS_BY_LETTER[word[0].upper()].append(word)
        phoneme_count[len(PHONEME_DICT[word][0])] += 1
        letter_count[len(word)] += 1   

    log_console_header("Words by Letter") 
    for letter in sorted(WORDS_BY_LETTER.keys()):
        logging.info("%s: %d words", letter, len(WORDS_BY_LETTER[letter]))

    log_console_header("Words by Phoneme Count")
    for count, num_words in sorted(phoneme_count.items()):
        logging.info("%d phonemes: %d words", count, num_words)

    log_console_header("Words by Letter Count")
    for count, num_words in sorted(letter_count.items()):
        logging.info("%d letters: %d words", count, num_words)
    
# --------------------------------    
# Calculate word averages
# --------------------------------

    # Write the word averages to a CSV file
    
    # Ask user if they want to calculate word averages
    calculate_averages = input("Calculate word averages? (y/n, default 'y'): ").strip().lower() or 'y'
    if calculate_averages == 'y':
        log_console_header("Calculating Word Averages")
        write_word_averages(PHONEME_DICT, get_phoneme_distance_dict())

    log_console_header("Beginning Best Set Search")
    # Ask user how many trials to run
    TRIALS = int(input("Enter the number of trials to run (default 100000): ") or "100000")
    TRIALS = max(TRIALS, 1)  # Ensure at least one trial

    best_scores, best_set, best_levenshtein, best_phoneme, best_shared = find_best_set_randomized(trials=TRIALS)

    def log_scores(score, list_name, words):
        logging.info("--------------------------------")
        logging.info("Best %s Set (%.6f):", list_name, score)
        logging.info("--------------------------------")
        for word in words:
            logging.info("%-12s  ->  %s", word.capitalize(), ' '.join(PHONEME_DICT[word][0]))

    # Log best levenshtein set
    log_scores(best_scores['best levenshtein'], "Levenshtein Distance", best_levenshtein)
    # Log best phoneme set
    log_scores(best_scores['best phoneme'], "Phoneme Distance", best_phoneme)
    # Log best shared set
    log_scores(best_scores['best shared'], "Shared Phoneme Sequence Count", best_shared)
    # Log best overall set
    log_scores(best_scores['score'], "Overall", best_set)

if __name__ == "__main__":
    main()