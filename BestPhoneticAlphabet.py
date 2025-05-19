import nltk
from nltk.corpus import cmudict
import editdistance
import random
import os
import csv
import math
import logging
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# 🔧 CONFIGURATION
FILTER_WORDS = True           # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
TRIALS = 10000                 # Number of random trials
MIN_PHONEME_LENGTH = 2          # Minimum number of phonemes required per word
MAX_PHONEME_LENGTH = 8         # Maximum number of phonemes allowed per word

PENALIZE_LENGTH_VARIANCE = True
LENGTH_VARIANCE_WEIGHT = 10  # Adjust this weight to control penalty severity

# Phoneme Coordinates
PHONEME_COORDINATES = {
    
    # VOWELS    
    # Vowel |  Backness (Front 0 / Central 0.5 / Back 1)     Height (Low [Open] 0 / Mid 0.5 / High [Close] 1)      Roundness (Rounded 0 / Unrounded 1)
    "AA":  (0,  1,      0,       0),    # ɑ
    "AE":  (0,  0,      0,       0),    # æ
    "AH":  (0,  0.5,    0.5,     0),    # ʌ or ə
    "AO":  (0,  1,      0.5,     1),    # ɔ
    "AW":  (0,  0.75,   0.5,     1),    # aʊ
    "AX":  (0,  0.5,    0.5,     0),    # ə (unstressed)
    "AY":  (0,  0.5,    0.5,     0),    # aɪ
    "EY":  (0,  0,      0.65,    0),    # e
    "EH":  (0,  0,      0.5,     0),    # ɛ
    "ER":  (0,  0.5,    0.5,     0),    # ɚ
    "IY":  (0,  0,      1,       0),    # i
    "IH":  (0,  0,      0.85,    0),    # ɪ
    "OW":  (0,  1,      0.65,    1),    # o
    "OY":  (0,  0.5,    0.5,     0.5),  # ɔɪ
    "UW":  (0,  1,      1,       1),    # u
    "UH":  (0,  1,      0.85,    1),    # ʊ
    
    # CONSONANTS
    # Consonant | Place of Articulation | Manner of Articulation | Voiced/Unvoiced
    # 0 = Bilabial, 0.2 = Labiodental, 0.4 = Dental, 0.6 = Alveolar, 0.8 = Velar, 1 = Glottal
    # 0 = Stop, 0.125 = Affricate, 0.25 = Fricative, 0.5 = Nasal, 0.75 = Lateral Liquid, 0.875 = Rhotic Liquid, 1 = Glide
    # 0 = Voiceless, 1 = Voiced
    
    # Stops
    "P":  (1, 0,    0,      0),  # voiceless bilabial stop
    "B":  (1, 0,    0,      1),  # voiced bilabial stop
    "D":  (1, 0.4,  0,      1),  # voiced alveolar stop
    "T":  (1, 0.4,  0,      0),  # voiceless alveolar stop
    "K":  (1, 0.8,  0,      0),  # voiceless velar stop
    "G":  (1, 0.8,  0,      1),  # voiced velar stop

    # Affricates``
    "CH": (1, 0.6,  0.125,  0),  # voiceless postalveolar affricate
    "JH": (1, 0.6,  0.125,  1),  # voiced postalveolar affricate

    # Fricatives
    "F":  (1, 0.2,  0.25,   0),  # voiceless labiodental fricative
    "V":  (1, 0.2,  0.25,   1),  # voiced labiodental fricative
    "TH": (1, 0.4,  0.25,   0),  # voiceless dental fricative
    "DH": (1, 0.4,  0.25,   1),  # voiced dental fricative
    "S":  (1, 0.4,  0.25,   0),  # voiceless alveolar fricative
    "Z":  (1, 0.4,  0.25,   1),  # voiced alveolar fricative
    "SH": (1, 0.6,  0.25,   0),  # voiceless postalveolar fricative
    "ZH": (1, 0.6,  0.25,   1),  # voiced postalveolar fricative
    "HH": (1, 1,    0.25,   0),  # voiceless glottal fricative

    # Nasals
    "M":  (1, 0,    0.5,    1),  # bilabial nasal
    "N":  (1, 0.4,  0.5,    1),  # alveolar nasal
    "NG": (1, 0.8,  0.5,    1),  # velar nasal

    # Liquids
    "L":  (1, 0.4,  0.75,   1),  # alveolar lateral liquid
    "R":  (1, 0.4,  0.875,  1),  # alveolar rhotic liquid

    # Glides (approximants)
    "Y":  (1, 0.6,  1,      1),  # palatal glide (IPA: /j/)
    "W":  (1, 0,    1,      1)   # bilabial glide
}   

SELECTED_FUNCTION = "phoneme_distance"  # Choose the distance function to use

PHONEME_MATRIX = []
PHONEME_DISTANCE_DICT = {}

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

    # If the lengths are different, add the distance for the remaining phonemes
    remaining_phonemes = {}
    if len(p1_list) > len(p2_list):
        remaining_phonemes = p1_list[len(p2_list):]
    else:
        remaining_phonemes = p2_list[len(p1_list):]
        
    distance += sum(PHONEME_DISTANCE_DICT.get((remaining_phonemes[i], ""), 0) for i in range(len(remaining_phonemes)))   
    return distance

DISTANCE_FUNC = {
    "levenshtein": levenshtein_distance,
    "phoneme_distance": phoneme_distance
}

def compute_phoneme_distance(p1, p2):
    distance_func = DISTANCE_FUNC.get(SELECTED_FUNCTION)
    return sum(distance_func(a, b) for a, b in itertools.zip_longest(p1, p2, fillvalue=""))

def compute_total_distance(word_set):
    distances = [
        compute_phoneme_distance(pron_dict[w1][0], pron_dict[w2][0])
        for w1, w2 in itertools.combinations(word_set, 2)
    ]
    base_score = sum(distances)

    penalty = 0
    if PENALIZE_LENGTH_VARIANCE:
        lengths = [len(pron_dict[w][0]) for w in word_set]
        std_dev = np.std(lengths)
        penalty = std_dev * LENGTH_VARIANCE_WEIGHT

    return base_score - penalty

def find_best_set_randomized(trials=1000): 
    headers = ["score", "algorithm"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    write_header = not os.path.exists("random_search_log.csv")
    
    """Find a diverse set via random sampling."""
    best_score = -1
    best_set = []
    letters = sorted(words_by_letter.keys())

    with open("random_search_log.csv", "a", newline="") as f:
        writer = csv.writer(f)  
        # Write header if file is new
        if write_header:
            writer.writerow(headers)

        for i in range(trials):
            candidate = [random.choice(words_by_letter[letter]) for letter in letters if words_by_letter[letter]]
            
            # Wait until we have a a full set of 26 letters
            if len(candidate) < 26:
                continue
            score = compute_total_distance(candidate)

            # Log this candidate and its score to the file
            # Sort the trial data
            candidate = sorted(candidate)
            # Write the trial data
            writer.writerow([score, "Random Search"] + candidate)
            
            # Check if this candidate is better than the best found so far
            if score > best_score:
                best_score = score
                best_set = candidate

            if (i + 1) % (trials // 10) == 0 or (i + 1) == trials:
                print(f"Progress: {((i + 1) / trials) * 100:.0f}%")

    return best_set, best_score

# Compare every phoneme coordinate with every other and write distances to a CSV.
def write_phoneme_distance_matrix(filename="phoneme_distances.csv"):
    phonemes = list(PHONEME_COORDINATES.keys())
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Phoneme1", "Phoneme2", "EuclideanDistance"])
        for i, p1 in enumerate(phonemes):
            coord1 = PHONEME_COORDINATES[p1]
            # Get magnitude of p1 coordinates and write to file
            magnitude = math.sqrt(sum(a ** 2 for a in coord1))
            
            writer.writerow([p1, "", magnitude])
            writer.writerow([p1, p1, 0.0])
            
            for j, p2 in enumerate(phonemes):
                if i < j:  # Avoid duplicate pairs and self-comparison
                    coord2 = PHONEME_COORDINATES[p2]
                    # Compute Euclidean distance
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                    writer.writerow([p1, p2, dist])
                    writer.writerow([p2, p1, dist])  # Symmetric distance

def load_phoneme_distance_matrix(filename="phoneme_distances.csv"):
    """Load the phoneme distance matrix from a CSV file."""
    matrix = []
    distance_dict = {}
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            p1, p2, dist = row[0], row[1], float(row[2])
            matrix.append([p1, p2, dist])
            distance_dict[(p1, p2)] = dist
            distance_dict[(p2, p1)] = dist  # Symmetric
    return matrix, distance_dict

def write_average_word_distances(pron_dict, filename="average_word_distances.csv"):
    """
    For each word, compare it to every other word (excluding words with the same first letter),
    compute the average phoneme distance, and write results to a CSV file.
    """
    words = list(pron_dict.keys())
    total_words = len(words)
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Word", "Magnitude", "AverageDistance"])
        for i, word1 in enumerate(words):     
            pron1 = pron_dict[word1][0] if pron_dict[word1] else []
            if not pron1:
                continue
            
            print(f"Processing word {i + 1}/{total_words}: {word1}")
            total_dist = 0.0
            count = 0
            
            for j, word2 in enumerate(words):
                if i == j or word2[0].upper() == word1[0].upper():
                    continue
                pron2 = pron_dict[word2][0] if pron_dict[word2] else []
                if not pron2:
                    continue
                dist = compute_phoneme_distance(pron1, pron2)
                total_dist += dist
                count += 1
                
            # Calculate the current word's phoneme magnitude I.E. The sum of all its phoneme distances
            magnitude = sum(PHONEME_DISTANCE_DICT.get((p, ""), 0) for p in pron1)
            avg_dist = total_dist / count if count else 0.0
            writer.writerow([word1, magnitude, avg_dist])

def get_cleaned_cmu_dict():
    pron_dict = cmudict.dict()
    print("\n🔄 CMU Dictionary Loaded\n--------------------------------\nTotal Words:", len(pron_dict))
    
    lemmatizer = WordNetLemmatizer()
    lemma_map = defaultdict(list)

    def lemmatize_word(word):
        noun = lemmatizer.lemmatize(word.lower(), pos='n')
        verb = lemmatizer.lemmatize(noun, pos='v')
        return verb

    for word in pron_dict:
        if not word.isalpha():
            continue
        lemma = lemmatize_word(word)
        lemma_map[lemma].append(word)

    # Choose representative word for each lemma (shortest spelling)
    cleaned_dict = {}
    for lemma, variants in lemma_map.items():
        representative = min(variants, key=len)
        cleaned_dict[representative] = pron_dict[representative]

    print("\n📦 CMU Pronouncing Dictionary Lemmatized\n--------------------------------\nTotal Words:", len(cleaned_dict))

    # Normalize the dictionary
    for word in list(cleaned_dict.keys()):
        # Remove words with non-alphabetic characters
        if not word.isalpha():
            del cleaned_dict[word]
        # Remove words with fewer than MIN_PHONEME_LENGTH
        elif len(cleaned_dict[word][0]) < MIN_PHONEME_LENGTH:
            del cleaned_dict[word]
        # Remove words with more than MAX_PHONEME_LENGTH
        elif len(cleaned_dict[word][0]) > MAX_PHONEME_LENGTH:
            del cleaned_dict[word]
        # Remove words that start with "ph", "sh", or "th"
        elif word.startswith("ph") or word.startswith("sh") or word.startswith("th"):
            del cleaned_dict[word]
        

    print("\n🔄 CMU Dictionary Normalization Complete\n--------------------------------\nTotal Words:", len(cleaned_dict))

    # Normalize phoneme representation
    for word, pronunciations in cleaned_dict.items():
        for i, pronunciation in enumerate(pronunciations):
            # Remove stress markers
            pronunciation = [p[:-1] if p[-1].isdigit() else p for p in pronunciation]
            # Convert to uppercase
            pronunciation = [p.upper() for p in pronunciation]
            cleaned_dict[word][i] = pronunciation
    print("\n🔄 Phoneme Normalization Complete\n--------------------------------")

    return cleaned_dict

# -----------------------------
# 📥 DATA LOADING & PREP
# -----------------------------

# Load CMU Pronouncing Dictionary
nltk.download('cmudict')
nltk.download("wordnet")

pron_dict = get_cleaned_cmu_dict()

# Group words by first letter
words_by_letter = defaultdict(list)
for word in pron_dict:
    if word[0].isalpha() and word.isalpha():
        first_letter = word[0].upper()
        words_by_letter[first_letter].append(word)

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------

# Pre-compute phoneme distances
# Load phoneme distance matrix
if not os.path.exists("phoneme_distances.csv"):
    write_phoneme_distance_matrix("phoneme_distances.csv")
    

PHONEME_MATRIX, PHONEME_DISTANCE_DICT = load_phoneme_distance_matrix("phoneme_distances.csv")

if not os.path.exists("average_word_distances.csv"):
    write_average_word_distances(pron_dict, "average_word_distances.csv")

best_by_letter = {}
with open("average_word_distances.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        word = row["Word"]
        avg_dist = float(row["AverageDistance"])
        if word and word[0].isalpha():
            letter = word[0].upper()
            if (letter not in best_by_letter) or (avg_dist > best_by_letter[letter][1]):
                best_by_letter[letter] = (word, avg_dist)

# Print the best word for each letter with phonemes
print("\n📋 Best Word for Each Letter (A–Z):")
for letter in sorted(best_by_letter.keys()):
    word, avg_dist = best_by_letter[letter]
    print(f"{letter}: {word} ({avg_dist:.3f})")
    #print(f"{letter}: {word} [{pron_dict[word][0]}] ({avg_dist:.3f})")

# pause for user input
input("Press Enter to continue...")

# Find optimal diverse word set (random trial or exhaustive)
print("🔍 Searching for the most phonetically diverse set of words...")
selected_words, score = find_best_set_randomized(trials=TRIALS)

# Print result
print("\n📋 Selected Words (Most Phonetically Diverse A–Z):")
for word in selected_words:
    print(f"{word.capitalize():<12}  ->  {' '.join(pron_dict[word][0])}")
print(f"\n🔢 Total Phoneme Distance Score: {score}")
