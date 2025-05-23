import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer
import editdistance
import random
import os
import csv
import math
import logging
import itertools
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import time

# 🔧 CONFIGURATION

# -----------------------------
# Filter Options
# -----------------------------
FILTER_NON_ALPHABETIC = True    # Set to 'True' to filter out words with non-alphabetic characters
FILTER_SINGLE_LETTER = True     # Set to 'True' to filter out single letter words
FILTER_BY_MIN_PHONEME = True     # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
FILTER_BY_MAX_PHONEME = True     # Set to 'True' to filter out words with more than MAX_PHONEME_LENGTH phonemes
FILTER_BY_MIN_LENGTH = False         # Set to 'True' to filter out words with fewer than MIN_WORD_LENGTH letters
FILTER_BY_MAX_LENGTH = False         # Set to 'True' to filter out words with more than MAX_WORD_LENGTH letters
FILTER_BY_MIN_SYLLABLES = True    # Set to 'True' to filter out words with fewer than MIN_SYLLABLES syllables
FILTER_BY_MAX_SYLLABLES = True   # Set to 'True' to filter out words with more than MAX_SYLLABLES syllables

# Prefix filters
FILTER_EU = True            # Set to 'True' to filter out words starting with "eu"
FILTER_GN = True            # Set to 'True' to filter out words starting with "gn"
FILTER_KN = True            # Set to 'True' to filter out words starting with "kn"
FILTER_MN = True            # Set to 'True' to filter out words starting with "mn"
FILTER_PH = True            # Set to 'True' to filter out words starting with "ph"
FILTER_PN = True            # Set to 'True' to filter out words starting with "pn"
FILTER_PS = True            # Set to 'True' to filter out words starting with "ps"
FILTER_SH = True            # Set to 'True' to filter out words starting with "sh"
FILTER_TH = True            # Set to 'True' to filter out words starting with "th"
FILTER_WR = True            # Set to 'True' to filter out words starting with "wr"

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

# We just don't fuck with these words. Slurs and racial remarks are a no-go, we diverse and tolerant in this sum-bitch.
# Mostly because 'X' only has a few words that makes sense
BANNED_WORDS = [
    "xhosa", "xian", "xinjiang", "xenophobia", "xenophobic", "xenophon", "iwo"
]

LEMMATIZE_DICT = True          # Set to 'True' to lemmatize words
CHECK_AGAINST_WORDNET = True   # Set to 'True' to check if words exist in WordNet

# -----------------------------
# Phoneme and Word Length Options
# -----------------------------
MIN_PHONEME_LENGTH = 2          # Minimum number of phonemes required per word
MAX_PHONEME_LENGTH = 10         # Maximum number of phonemes allowed per word
MIN_WORD_LENGTH = 3             # Minimum number of letters required per word
MAX_WORD_LENGTH = 10            # Maximum number of letters allowed per word
MIN_SYLLABLES = 2                # Minimum number of syllables allowed per word
MAX_SYLLABLES = 3               # Maximum number of syllables allowed per word

# -----------------------------
# Randomization Options
# -----------------------------
TRIALS = 10000                  # Number of random trials
PENALIZE_LENGTH_VARIANCE = True # Set to 'True' to penalize length variance
LENGTH_VARIANCE_WEIGHT = 10     # Adjust this weight to control penalty severity

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
PHONEME_DICT = {}
PHONEME_DISTANCE_DICT = {}
WORDS_BY_LETTER = defaultdict(list)

SIMILAR_PHONEME_PAIRS = {
    # Similar phoneme pairs
    ["AA", "AE"], # father, cat
    ["AH", "AO"], # cut, caught
    ["AO", "AW"], # caught, cow
    ["EY", "EH"], # they, bed
    ["IY", "IH"], # see, sit
    ["UW", "UH"], # too, put
}

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

# Sums the Levenshtein distances between all pairs of words in a set.
def total_levenshtein_distance(word_set):
    distances = [
        levenshtein_distance(w1, w2)
        for w1, w2 in itertools.combinations(word_set, 2)
    ]
    base_score = sum(distances)

    penalty = 0
    if PENALIZE_LENGTH_VARIANCE:
        lengths = [len(w) for w in word_set]
        std_dev = np.std(lengths)
        penalty = std_dev * LENGTH_VARIANCE_WEIGHT

    return base_score - penalty
    
# Compute the total distance score for a set of words.
def total_phoneme_distance(word_set):
    distances = [
        phoneme_distance(PHONEME_DICT[w1][0], PHONEME_DICT[w2][0])
        for w1, w2 in itertools.combinations(word_set, 2)
    ]

    return sum(distances)

# Compute the total shared phoneme distance for a set of words.
def total_shared_phoneme_distance(word_set):
    distances = [
        shared_phoneme_sequences(PHONEME_DICT[w1][0], PHONEME_DICT[w2][0])
        for w1, w2 in itertools.combinations(word_set, 2)
    ]

    return sum(distances)

# Find the best set of words via random sampling.
def find_best_set_randomized(trials=1000): 
    
    """Find a diverse set via random sampling."""
    best_score = -1
    best_score_levenshtein = -1
    best_score_phoneme = -1
    best_score_shared = 99999999999
    
    best_set = []
    best_levenshtein = []
    best_phoneme = []
    best_shared = []
    
    letters = sorted(WORDS_BY_LETTER.keys())

    # Delete the file if it exists
    if os.path.exists("random_search_log.csv"):
        os.remove("random_search_log.csv")

    with open("random_search_log.csv", "a", newline="") as f:
        writer = csv.writer(f)  
        
        # Write header if file is new
        writer.writerow(["Score", "Avg Levenshtein Distance", "Avg Phoneme Distance (Basic)", "Avg Shared Phoneme Sequence Count"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

        for i in range(trials):
            candidate = [random.choice(WORDS_BY_LETTER[letter]) for letter in letters if WORDS_BY_LETTER[letter]]
            
            # Wait until we have a a full set of 26 letters
            if len(candidate) < 26:
                continue
            
            # Log this candidate and its score to the file
            candidate = sorted(candidate)
            
            # Calculate the average distances 
            avg_levenshtein = total_levenshtein_distance(candidate) / len(candidate)
            avg_phoneme =   total_phoneme_distance(candidate) / len(candidate)
            avg_shared = total_shared_phoneme_distance(candidate) / len(candidate)
            score = avg_levenshtein + avg_phoneme - avg_shared

            # Write the trial data
            writer.writerow([score, avg_levenshtein, avg_phoneme, avg_shared] + candidate)
            
            # Check if this candidate is better than the best found so far
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

            if (i + 1) % (trials // 25) == 0 or (i + 1) == trials:
                print(f"Progress: {((i + 1) / trials) * 100:.0f}%")

    best_scores = {
        "score": best_score,
        "best levenshtein": best_score_levenshtein,
        "best phoneme": best_score_phoneme,
        "best shared": best_score_shared
    }
    

    return best_scores, best_set, best_levenshtein, best_phoneme, best_shared 

# Write the word averages to a CSV file.
def write_word_averages(pron_dict):
    with open("word_score_averages.csv", "w", newline="") as avg_csvfile:
        avg_writer = csv.writer(avg_csvfile)
        avg_writer.writerow(["Word", "Phonemes", "Magnitude", "Average Levenshtein Distance", "Average Phoneme Distance (Basic)", "Average Shared Phoneme Sequence Count", "Calculation Time"])

        all_magnitudes = 0
        all_levenshtein = 0
        all_phoneme = 0
        all_shared = 0
        all_time = 0
        
        # For each word, compare it to every other word (excluding words with the same first letter),
        # compute the average Levenshtein distance, and write results to a CSV file.
        total_words = len(list(pron_dict.keys()))
        for i, word1 in enumerate(list(pron_dict.keys())):
            print(f"Processing word {i + 1}/{total_words}: {word1}")
            
            start_time = time.time()
            
            word1_pron = pron_dict[word1][0]
            
            total_dist_levenshtein = 0.0
            total_dist_phoneme = 0.0
            total_shared = 0.0
            
            count = 0
            
            # Go through check_dict and calculate the distance
            # between word1 and every other word in the dictionary
            for j, word2 in enumerate(list(pron_dict.keys())):
                if i == j or word2[0].upper() == word1[0].upper():
                    continue
                
                word2_pron = pron_dict[word2][0]

                total_dist_levenshtein += levenshtein_distance(word1, word2)
                total_dist_phoneme += sum(phoneme_distance(a,b) for a,b in itertools.zip_longest(word1_pron, word2_pron, fillvalue=""))   
                total_shared += shared_phoneme_sequences(word1_pron, word2_pron)

                count += 1
            
            magnitude = sum(PHONEME_DISTANCE_DICT.get((p, ""), 0) for p in word1_pron) / len(word1_pron)
            avg_levenshtein = total_dist_levenshtein / count            
            avg_phoneme = total_dist_phoneme / count          
            avg_shared = total_shared / count
            compute_time = time.time() - start_time

            all_magnitudes += magnitude
            all_levenshtein += avg_levenshtein
            all_phoneme += avg_phoneme
            all_shared += avg_shared
            all_time += compute_time
            
            # Write the results to the CSV file
            word1_pron = " ".join(word1_pron)
            avg_writer.writerow([word1, word1_pron, magnitude, avg_levenshtein, avg_phoneme, avg_shared, compute_time])         
        
        # Write the averages to the CSV file
        all_magnitudes /= total_words
        all_levenshtein /= total_words
        all_phoneme /= total_words
        all_shared /= total_words
        all_time /= total_words
        avg_writer.writerow(["Total Averages", "", all_magnitudes, all_levenshtein, all_phoneme, all_shared, all_time])
        
# Clean the CMU Pronouncing Dictionary and apply filters.
def get_cleaned_cmu_dict():
    print("\nCMU Dictionary Loaded | Total Words:", len(cmudict.dict()))

    filters_list = [
        FILTER_SINGLE_LETTER,
        FILTER_BY_MIN_PHONEME,
        FILTER_BY_MAX_PHONEME,
        FILTER_BY_MIN_LENGTH,
        FILTER_BY_MAX_LENGTH,
        CHECK_AGAINST_WORDNET,
        LEMMATIZE_DICT
    ]

    # Check if any filters are active
    if not any(filters_list):
        print("\nNo Filters Active | All Words Kept")
    else:
        print("\n - [Filters Active] - \n")
        # Print the filters that are active
        if FILTER_NON_ALPHABETIC:
            print(" - [Words With Non-Alphabetic Characters]")
        if FILTER_SINGLE_LETTER:
            print(" - [Single Letter Words]")
        if FILTER_BY_MIN_PHONEME:
            print(" - [Min Phoneme Length (", MIN_PHONEME_LENGTH, ")")
        if FILTER_BY_MAX_PHONEME:
            print(" - [Max Phoneme Length (", MAX_PHONEME_LENGTH, ")")
        if FILTER_BY_MIN_LENGTH:
            print(" - [Min Word Length (",MIN_WORD_LENGTH, ")")
        if FILTER_BY_MAX_LENGTH:
            print(" - [Max Word Length (", MAX_WORD_LENGTH, ")")
        if CHECK_AGAINST_WORDNET:
            print(" - [WordNet Check]")
        if LEMMATIZE_DICT:
            print(" - [Lemmatization]")

    cleaned_dict = cmudict.dict()
    
    # Normalize the dictionary
    for word in list(cleaned_dict.keys()):
        # Remove words with non-alphabetic characters
        if FILTER_NON_ALPHABETIC and not word.isalpha():
            del cleaned_dict[word]
        # Remove banned words
        elif word in BANNED_WORDS:
            del cleaned_dict[word]    
        # Remove words that contain any of the specified prefixes
        elif any(word.startswith(prefix) for prefix, active in PREFIX_FILTERS.items() if active):
            del cleaned_dict[word]
        # Remove words with fewer than MIN_PHONEME_LENGTH
        elif FILTER_BY_MIN_PHONEME and len(cleaned_dict[word][0]) < MIN_PHONEME_LENGTH:
            del cleaned_dict[word]
        # Remove words with more than MAX_PHONEME_LENGTH
        elif FILTER_BY_MAX_PHONEME and len(cleaned_dict[word][0]) > MAX_PHONEME_LENGTH:
            del cleaned_dict[word]
        # Remove words that less than 2 letters long
        elif FILTER_BY_MIN_LENGTH and len(word) < MIN_WORD_LENGTH:
            del cleaned_dict[word]
        # Remove words that are longer than 15 letters
        elif FILTER_BY_MAX_LENGTH and len(word) > MAX_WORD_LENGTH:
            del cleaned_dict[word]
        # Remove words that are made up of a single letter repeated
        elif FILTER_SINGLE_LETTER and len(set(word)) == 1:
            del cleaned_dict[word]
        # Remove words that have a banned phoneme prefix
        elif any(word.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(word[0].lower(), [])):
            del cleaned_dict[word]
        # Remove words that are not in WordNet (if CHECK_AGAINST_WORDNET is True) (Results in about 25k words left, 80k without checking)
        elif CHECK_AGAINST_WORDNET and not wordnet.synsets(word):
            del cleaned_dict[word]
        # MUST BE AFTER EVERYTHING | Remove words with less/more than MIN/MAX Syllables
        elif FILTER_BY_MIN_SYLLABLES or FILTER_BY_MAX_SYLLABLES:
            syllable_count = len([p for p in cleaned_dict[word][0] if p[-1].isdigit()])
            if syllable_count < MIN_SYLLABLES or syllable_count > MAX_SYLLABLES:
                del cleaned_dict[word]
                
    print("\nCulling Complete | Total Words:", len(cleaned_dict))

    if LEMMATIZE_DICT:
        print("\nLemmatizing Words...")
        
        # Create a lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemma_map = defaultdict(list)
        
        # Lemmatize the words, group by lemma
        for word, prons in cleaned_dict.items():
            noun = lemmatizer.lemmatize(word.lower(), pos='n')
            lemma = lemmatizer.lemmatize(noun, pos='v')
            lemma_map[lemma].append((word, prons[0]))

        temp_dict = {}
        # Choose representative word for each lemma (shortest spelling)
        for lemma, variants in lemma_map.items():
            representative = min(variants, key=lambda x: len(x[0]))
            temp_dict[representative[0]] = [representative[1]]  # keep as list for CMU compatibility 
        
        cleaned_dict = temp_dict
        print("\nLemmatization Complete | Total Words:", len(cleaned_dict))

    print("\nCleaning Phoneme Representation")
    print("\n - [Removing stress markers and numbers for simplicity]")
    # Normalize phoneme representation
    for word, pronunciations in cleaned_dict.items():
        for i, pronunciation in enumerate(pronunciations):
            # Remove stress markers
            pronunciation = [p[:-1] if p[-1].isdigit() else p for p in pronunciation]
            # Convert to uppercase
            pronunciation = [p.upper() for p in pronunciation]
            cleaned_dict[word][i] = pronunciation
    
    print("\nPhoneme Normalization Complete | Total Words:", len(cleaned_dict))

    return cleaned_dict

# -----------------------------
#  DATA LOADING & PREP
# -----------------------------

# ============================
# Prep Phoneme Distance Dictionary
# ============================

# Compare every phoneme coordinate with every other, as well as itself and an empty string
phonemes = list(PHONEME_COORDINATES.keys())
for i, p1 in enumerate(phonemes):
    coord1 = PHONEME_COORDINATES[p1]
    # Get magnitude of p1 coordinates and write to file
    magnitude = math.sqrt(sum(a ** 2 for a in coord1))
    
    PHONEME_DISTANCE_DICT[(p1, "")] = magnitude
    PHONEME_DISTANCE_DICT[("", p1)] = magnitude
    PHONEME_DISTANCE_DICT[(p1, p1)] = 0.0
    
    for j, p2 in enumerate(phonemes):
        if i < j:  # Avoid duplicate pairs
            coord2 = PHONEME_COORDINATES[p2]
            # Compute Euclidean distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
            PHONEME_DISTANCE_DICT[(p1, p2)] = dist
            PHONEME_DISTANCE_DICT[(p2, p1)] = dist  

# --------------------------------
# Prep Dictionary
# --------------------------------

# Load NLTK
# Load CMU Pronouncing Dictionary
nltk.download('cmudict')
nltk.download("wordnet")

PHONEME_DICT = get_cleaned_cmu_dict()

WORDS_BY_LETTER = defaultdict(list)
# Group words by first letter
for word in PHONEME_DICT.keys():
    WORDS_BY_LETTER[word[0].upper()].append(word)

# Print the number of words for each letter
print("\nWords by Letter")
for letter in sorted(WORDS_BY_LETTER.keys()):
    print(f"{letter}: {len(WORDS_BY_LETTER[letter])} words")

# Print total number of words per number of phonemes
print("\nWords by Number of Phonemes")
phoneme_count = defaultdict(int)
for word in PHONEME_DICT.keys():
    phoneme_count[len(PHONEME_DICT[word][0])] += 1
for count, num_words in sorted(phoneme_count.items()):
    print(f"{count} phonemes: {num_words} words")

# Print total number of words per number of letters
print("\nWords by Length")
letter_count = defaultdict(int)
for word in PHONEME_DICT.keys():
    letter_count[len(word)] += 1
for count, num_words in sorted(letter_count.items()):
    print(f"{count} letters: {num_words} words")

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------

#write_word_averages(PHONEME_DICT)

best_scores, best_set, best_levenshtein, best_phoneme, best_shared = find_best_set_randomized(trials=TRIALS)

# Print best levenshtein set
print(f"\nBest Levenshtein Set ({best_scores['best levenshtein']:.6f}):")
for word in best_levenshtein:
    print(f"{word.capitalize():<12}  ->  {' '.join(PHONEME_DICT[word][0])}")

# Print best phoneme set
print(f"\nBest Phoneme Distance Set ({best_scores['best phoneme']:.6f}):")
for word in best_phoneme:
    print(f"{word.capitalize():<12}  ->  {' '.join(PHONEME_DICT[word][0])}")

# Print best shared set
print(f"\nBest Shared Phoneme Sequence Count Set ({best_scores['best shared']:.6f}):")
for word in best_shared:
    print(f"{word.capitalize():<12}  ->  {' '.join(PHONEME_DICT[word][0])}")

# Print best overall set
print("\nBest Overall Set:")
for word in best_set:
    print(f"{word.capitalize():<12}  ->  {' '.join(PHONEME_DICT[word][0])}")