import nltk
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

# 🔧 CONFIGURATION
FILTER_WORDS = False           # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
MAX_WORDS_PER_LETTER = 1000000   # Max candidate words per starting letter
TRIALS = 10000                 # Number of random trials
MIN_PHONEME_LENGTH = 1        # Minimum number of phonemes required per word
VALIDATE = True              # Set to 'True' to run validation checks on the CMU dictionary
USE_FEATURE_BASED_DISTANCE = False  # Set to 'True' to use feature-based distance instead of Levenshtein

PENALIZE_LENGTH_VARIANCE = True
LENGTH_VARIANCE_WEIGHT = 10  # Adjust this weight to control penalty severity

# Phoneme Coordinates
PHONEME_COORDINATES = {
    
    # VOWELS    
    # Vowel |  Backness (Front 0 / Central 1 / Back 2)     Height (Low [Open] 0 / Mid 1 / High [Close] 2)      Roundness (Rounded 0 / Unrounded 1)
    "AA":  (0,  2,   0, 0),     # ɑ
    "AE":  (0,  0,   0, 0),     # æ
    "AH":  (0,  1,   1, 0),     # ʌ or ə
    "AO":  (0,  2,   1, 1),     # ɔ
    "AW":  (0,  1.5, 1, 1),     # aʊ
    "AX":  (0,  1,   1, 0),     # ə (unstressed)
    "AY":  (0,  1,   1, 0),     # aɪ
    "EY":  (0,  0,   1.3, 0),   # e
    "EH":  (0,  0,   1, 0),     # ɛ
    "ER":  (0,  1,   1, 0),     # ɚ
    "IY":  (0,  0,   2, 0),     # i
    "IH":  (0,  0,   1.7, 0),   # ɪ
    "OW":  (0,  2,   1.3, 1),   # o
    "OY":  (0,  1,   1, 0.5),   # ɔɪ
    "UW":  (0,  2,   2, 1),     # u
    "UH":  (0,  2,   1.7, 1),   # ʊ
    
    
    # CONSONANTS    
    # 
    # 
    # Consonant | Place of Articulation | Manner of Articulation | Voiced/Unvoiced
    # 0 = Bilabial, 1 = Labiodental, 2 = Dental, 3 = Alveolar, 4 = Velar, 5 = Glottal
    # 0 = Stop, 0.5 = Affricate, 1 = Fricative, 2 = Nasal, 3 = Lateral Liquid, 3.5 = Rhotic Liquid, 4 = Glide
    # 0 = Voiceless, 1 = Voiced
    
    # Stops
    "P":  (1, 0, 0, 0),  # voiceless bilabial stop
    "B":  (1, 0, 0, 1),  # voiced bilabial stop
    "T":  (1, 2, 0, 0),  # voiceless alveolar stop
    "D":  (1, 2, 0, 1),  # voiced alveolar stop
    "K":  (1, 4, 0, 0),  # voiceless velar stop
    "G":  (1, 4, 0, 1),  # voiced velar stop

    # Affricates
    "CH": (1, 3, 0.5, 0),  # voiceless postalveolar affricate
    "JH": (1, 3, 0.5, 1),  # voiced postalveolar affricate

    # Fricatives
    "F":  (1, 1, 1, 0),  # voiceless labiodental fricative
    "V":  (1, 1, 1, 1),  # voiced labiodental fricative
    "TH": (1, 2, 1, 0),  # voiceless dental fricative
    "DH": (1, 2, 1, 1),  # voiced dental fricative
    "S":  (1, 2, 1, 0),  # voiceless alveolar fricative
    "Z":  (1, 2, 1, 1),  # voiced alveolar fricative
    "SH": (1, 3, 1, 0),  # voiceless postalveolar fricative
    "ZH": (1, 3, 1, 1),  # voiced postalveolar fricative
    "HH": (1, 5, 1, 0),  # voiceless glottal fricative

    # Nasals
    "M":  (1, 0, 2, 1),  # bilabial nasal
    "N":  (1, 2, 2, 1),  # alveolar nasal
    "NG": (1, 4, 2, 1),  # velar nasal

    # Liquids
    "L":  (1, 2, 3, 1),  # alveolar lateral liquid
    "R":  (1, 2, 3.5, 1),# alveolar rhotic liquid

    # Glides (approximants)
    "Y":  (1, 3, 4, 1),  # palatal glide (IPA: /j/)
    "W":  (1, 0, 4, 1)   # bilabial glide
}   


# -----------------------------
# 📦 FUNCTION DEFINITIONS
# -----------------------------

# This is the naive attempt to calculate the phoneme distance between two words
def phoneme_distance_levenshtein(w1, w2):
    """Calculate Levenshtein distance between two words' primary phoneme lists."""
    p1 = pron_dict[w1][0]
    p2 = pron_dict[w2][0]
    return editdistance.eval(p1, p2)

def compute_phoneme_distance(p1, p2):
    distance_func = phoneme_distance_levenshtein
    return sum(distance_func(a, b) for a, b in itertools.zip_longest(p1, p2, fillvalue=""))

def compute_total_distance(word_set):
    distances = [
        compute_phoneme_distance(w1[0], w2[0])
        for w1, w2 in itertools.combinations(word_set, 2)
    ]
    base_score = sum(distances)

    if PENALIZE_LENGTH_VARIANCE:
        lengths = [len(pron_dict[w][0]) for w in word_set]
        std_dev = np.std(lengths)
        penalty = std_dev * LENGTH_VARIANCE_WEIGHT
        return base_score - penalty  # subtracting penalty lowers the score
    else:
        return base_score

def find_best_set_randomized(words_by_letter, trials=1000): 
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



def write_phoneme_distance_matrix(filename="phoneme_distances.csv"):
    """Compare every phoneme coordinate with every other and write distances to a CSV."""
    phonemes = list(PHONEME_COORDINATES.keys())
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Phoneme1", "Phoneme2", "EuclideanDistance"])
        for i, p1 in enumerate(phonemes):
            coord1 = PHONEME_COORDINATES[p1]
            for j, p2 in enumerate(phonemes):
                if i < j:  # Avoid duplicate pairs and self-comparison
                    coord2 = PHONEME_COORDINATES[p2]
                    # Compute Euclidean distance
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                    writer.writerow([p1, p2, dist])


# -----------------------------
# 📥 DATA LOADING & PREP
# -----------------------------

nltk.download('cmudict')
from nltk.corpus import cmudict
pron_dict = cmudict.dict()

# Group words by first letter
words_by_letter = defaultdict(list)
for word in pron_dict:
    if word[0].isalpha() and word.isalpha():
        first_letter = word[0].upper()
        words_by_letter[first_letter].append(word)

# Filter and trim word lists by minimum phoneme length (if desired)
if FILTER_WORDS:
    for letter in words_by_letter:
        filtered = [w for w in words_by_letter[letter] if len(pron_dict[w][0]) >= MIN_PHONEME_LENGTH]
        words_by_letter[letter] = filtered[:MAX_WORDS_PER_LETTER]


# Validate the CMU dictionary (if desired)
# Validate CMU dictionary
# Goes through the CMU dictionary and checks for:
# 1. Word counts by starting letter
# 2. Phoneme length distribution
# 3. Check if any letters are missing or underpopulated
# 4. Example problematic or variant words
# 5. Total word count
if VALIDATE:
    print("\n🔍 CMU Dictionary Validation Report\n")
    
    # 0. Total word count
    print(f"\n📦 Total Words in CMU Dictionary: {len(pron_dict)}")
    
    # 1. Word counts by starting letter
    print("📊 Word Count by First Letter (filtered):")
    for letter in sorted(words_by_letter.keys()):
        count = len(words_by_letter[letter])
        print(f"  {letter}: {count} words")

    # 2. Get the total number of each phoneme
    phoneme_counter = Counter()

    for word, pronunciations in pron_dict.items():
        for phoneme_list in pronunciations:
            phoneme_counter.update(phoneme_list)

    # Sort phoneme counts
    phoneme_counter = dict(sorted(phoneme_counter.items(), key=lambda item: item[1], reverse=True))
    
    # Print phoneme counts        
    print("\n🔢 Phoneme Totals in CMU Dictionary:")
    for phoneme, count in phoneme_counter.items():
        print(f"{phoneme:<5} : {count}")

    # 3. Phoneme length distribution
    phoneme_lengths = Counter()
    for word in pron_dict:
        try:
            phoneme_lengths[len(pron_dict[word][0])] += 1
        except IndexError:
            continue

    print("\n📏 Phoneme Length Distribution (first pronunciation only):")
    for length, count in sorted(phoneme_lengths.items()):
        print(f"  {length} phonemes: {count} words")

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------


# write_phoneme_distance_matrix("phoneme_distances.csv")

# Find optimal diverse word set (random trial or exhaustive)
#print("🔍 Searching for the most phonetically diverse set of words...")
#selected_words, score = find_best_set_randomized(words_by_letter, trials=TRIALS)

# Print result
#print("\n📋 Selected Words (Most Phonetically Diverse A–Z):")
#for word in selected_words:
#    print(f"{word.capitalize():<12}  ->  {' '.join(pron_dict[word][0])}")
#print(f"\n🔢 Total Phoneme Distance Score: {score}")





# -----------------------------
# 📈 GRAPH VISUALIZATION
# -----------------------------

# Build graph
# G = nx.Graph()
# for word in selected_words:
#     G.add_node(word)
# 
# for w1, w2 in combinations(selected_words, 2):
#     dist = compute_phoneme_distance(w1, w2)
#     G.add_edge(w1, w2, weight=dist)
# 
# # Draw graph
# plt.figure(figsize=(15, 12))
# pos = nx.spring_layout(G, weight='weight', seed=42)
# nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
# plt.title("Phoneme Dissimilarity Graph for Selected Words")
# plt.show()