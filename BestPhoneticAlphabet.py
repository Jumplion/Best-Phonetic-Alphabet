import nltk
import editdistance
import random
import os
import csv
import logging
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations

# 🔧 CONFIGURATION
FILTER_WORDS = False           # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
MAX_WORDS_PER_LETTER = 50000   # Max candidate words per starting letter
TRIALS = 100000                 # Number of random trials
MIN_PHONEME_LENGTH = 0        # Minimum number of phonemes required per word
VALIDATE = True              # Set to 'True' to run validation checks on the CMU dictionary
USE_FEATURE_BASED_DISTANCE = False  # Set to 'True' to use feature-based distance instead of Levenshtein

PENALIZE_LENGTH_VARIANCE = True
LENGTH_VARIANCE_WEIGHT = 10  # Adjust this weight to control penalty severity

arpabet_features = {
    'P': {'voiced': 0, 'place': 'bilabial', 'manner': 'stop'},
    'B': {'voiced': 1, 'place': 'bilabial', 'manner': 'stop'},
    'T': {'voiced': 0, 'place': 'alveolar', 'manner': 'stop'},
    'D': {'voiced': 1, 'place': 'alveolar', 'manner': 'stop'},
    'K': {'voiced': 0, 'place': 'velar', 'manner': 'stop'},
    'G': {'voiced': 1, 'place': 'velar', 'manner': 'stop'},
    'CH': {'voiced': 0, 'place': 'postalveolar', 'manner': 'affricate'},
    'JH': {'voiced': 1, 'place': 'postalveolar', 'manner': 'affricate'},
    'F': {'voiced': 0, 'place': 'labiodental', 'manner': 'fricative'},
    'V': {'voiced': 1, 'place': 'labiodental', 'manner': 'fricative'},
    'TH': {'voiced': 0, 'place': 'dental', 'manner': 'fricative'},
    'DH': {'voiced': 1, 'place': 'dental', 'manner': 'fricative'},
    'S': {'voiced': 0, 'place': 'alveolar', 'manner': 'fricative'},
    'Z': {'voiced': 1, 'place': 'alveolar', 'manner': 'fricative'},
    'SH': {'voiced': 0, 'place': 'postalveolar', 'manner': 'fricative'},
    'ZH': {'voiced': 1, 'place': 'postalveolar', 'manner': 'fricative'},
    'HH': {'voiced': 0, 'place': 'glottal', 'manner': 'fricative'},
    'M': {'voiced': 1, 'place': 'bilabial', 'manner': 'nasal'},
    'N': {'voiced': 1, 'place': 'alveolar', 'manner': 'nasal'},
    'NG': {'voiced': 1, 'place': 'velar', 'manner': 'nasal'},
    'L': {'voiced': 1, 'place': 'alveolar', 'manner': 'liquid'},
    'R': {'voiced': 1, 'place': 'alveolar', 'manner': 'liquid'},
    'Y': {'voiced': 1, 'place': 'palatal', 'manner': 'glide'},
    'W': {'voiced': 1, 'place': 'bilabial', 'manner': 'glide'},
    # Vowels (simplified)
    'AA': {'height': 'low', 'backness': 'back', 'rounded': 0},
    'AE': {'height': 'low', 'backness': 'front', 'rounded': 0},
    'AH': {'height': 'mid', 'backness': 'central', 'rounded': 0},
    'AO': {'height': 'mid', 'backness': 'back', 'rounded': 1},
    'AW': {'height': 'low', 'backness': 'back', 'rounded': 1},
    'AY': {'height': 'low', 'backness': 'front', 'rounded': 0},
    'EH': {'height': 'mid', 'backness': 'front', 'rounded': 0},
    'ER': {'height': 'mid', 'backness': 'central', 'rounded': 0},
    'EY': {'height': 'mid', 'backness': 'front', 'rounded': 0},
    'IH': {'height': 'high', 'backness': 'front', 'rounded': 0},
    'IY': {'height': 'high', 'backness': 'front', 'rounded': 0},
    'OW': {'height': 'mid', 'backness': 'back', 'rounded': 1},
    'OY': {'height': 'mid', 'backness': 'back', 'rounded': 1},
    'UH': {'height': 'high', 'backness': 'back', 'rounded': 1},
    'UW': {'height': 'high', 'backness': 'back', 'rounded': 1}
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

def feature_distance(p1, p2):
    base1, base2 = p1.strip("012"), p2.strip("012")
    f1, f2 = arpabet_features.get(base1), arpabet_features.get(base2)
    if not f1 or not f2:
        return 1  # default distance
    return sum(abs(f1[k] - f2[k]) if isinstance(f1[k], int) else int(f1[k] != f2[k]) for k in f1)

def compute_phoneme_distance(p1, p2):
    distance_func = feature_distance if USE_FEATURE_BASED_DISTANCE else phoneme_distance_levenshtein
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

# Find optimal diverse word set (random trial or exhaustive)
print("🔍 Searching for the most phonetically diverse set of words...")
selected_words, score = find_best_set_randomized(words_by_letter, trials=TRIALS)

# Print result
print("\n📋 Selected Words (Most Phonetically Diverse A–Z):")
for word in selected_words:
    print(f"{word.capitalize():<12}  ->  {' '.join(pron_dict[word][0])}")
print(f"\n🔢 Total Phoneme Distance Score: {score}")





# -----------------------------
# 📈 GRAPH VISUALIZATION
# -----------------------------

# Build graph
G = nx.Graph()
for word in selected_words:
    G.add_node(word)

for w1, w2 in combinations(selected_words, 2):
    dist = compute_phoneme_distance(w1, w2)
    G.add_edge(w1, w2, weight=dist)

# Draw graph
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, weight='weight', seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
plt.title("Phoneme Dissimilarity Graph for Selected Words")
plt.show()
