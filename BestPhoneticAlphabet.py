import nltk
import networkx as nx
import matplotlib.pyplot as plt
import editdistance
from collections import Counter, defaultdict
from itertools import combinations
import random
import os
import csv

# 🔧 CONFIGURATION
FILTER_WORDS = False           # Set to 'True' to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
MAX_WORDS_PER_LETTER = 10000   # Max candidate words per starting letter
TRIALS = 10000                 # Number of random trials
MIN_PHONEME_LENGTH = 0        # Minimum number of phonemes required per word
USE_RANDOM_SEARCH = True      # Set to 'True' to use random sampling, False to use greedy exhaustive search
VALIDATE = True              # Set to 'True' to run validation checks on the CMU dictionary

# -----------------------------
# 📦 FUNCTION DEFINITIONS
# -----------------------------

def phoneme_distance(w1, w2):
    """Calculate Levenshtein distance between two words' primary phoneme lists."""
    p1 = pron_dict[w1][0]
    p2 = pron_dict[w2][0]
    return editdistance.eval(p1, p2)

def phoneme_distance_average(w1, w2):
    """Calculate average phoneme distance between two words' phoneme lists."""
    p1 = pron_dict[w1]
    p2 = pron_dict[w2]
    distances = [editdistance.eval(p1[i], p2[j]) for i in range(len(p1)) for j in range(len(p2))]
    return sum(distances) / len(distances) if distances else float('inf')

def total_distance(words):
    """Compute total pairwise phoneme distances within a word set."""
    return sum(phoneme_distance(w1, w2) for w1, w2 in combinations(words, 2))

def find_best_set_randomized(words_by_letter, trials=1000):
    """Find a diverse set via random sampling."""
    best_score = -1
    best_set = []
    letters = sorted(words_by_letter.keys())

    for i in range(trials):
        candidate = [random.choice(words_by_letter[letter]) for letter in letters if words_by_letter[letter]]
        
        # Wait until we have a a full set of 26 letters
        if len(candidate) < 26:
            continue
        score = total_distance(candidate)

        # Log this candidate and its score to the file
        log_trial_to_csv("random_search_log.csv", score, candidate, "Random Search")
        
        # Check if this candidate is better than the best found so far
        if score > best_score:
            best_score = score
            best_set = candidate

        if (i + 1) % (trials // 10) == 0 or (i + 1) == trials:
            print(f"Progress: {((i + 1) / trials) * 100:.0f}%")

    return best_set, best_score

def find_best_set_exhaustive(words_by_letter):
    """Greedy algorithm that selects one word per letter, maximizing dissimilarity at each step."""
    selected = []
    letters = sorted(words_by_letter.keys())

    for letter in letters:
        best_word = None
        best_min_dist = -1
        candidates = words_by_letter[letter]

        for candidate in candidates:
            if not selected:
                best_word = candidate
                break
            min_dist = min(phoneme_distance(candidate, existing) for existing in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_word = candidate

        if best_word:
            selected.append(best_word)

    return selected, total_distance(selected)

def log_trial_to_csv(csv_path, score, trial, algorithm_name):
    headers = ["score", "algorithm"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)  
        # Write header if file is new
        if write_header:
            writer.writerow(headers)

        # Sort the trial data
        trial = sorted(trial)
        # Write the trial data
        writer.writerow([score, algorithm_name] + trial)


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

    # 1. Word counts by starting letter
    print("📊 Word Count by First Letter (filtered):")
    for letter in sorted(words_by_letter.keys()):
        count = len(words_by_letter[letter])
        print(f"  {letter}: {count} words")

    # 2. Phoneme length distribution
    phoneme_lengths = Counter()
    for word in pron_dict:
        try:
            phoneme_lengths[len(pron_dict[word][0])] += 1
        except IndexError:
            continue

    print("\n📏 Phoneme Length Distribution (first pronunciation only):")
    for length, count in sorted(phoneme_lengths.items()):
        print(f"  {length} phonemes: {count} words")

    print("\n📏 Phoneme Length Distribution (all pronunciations):")
    for word, pron in pron_dict.items():
        for p in pron:
            phoneme_lengths[len(p)] += 1
    for length, count in sorted(phoneme_lengths.items()):
        print(f"  {length} phonemes: {count} words")
        

    # 3. Example problematic or variant words
    print("\n🧬 Sample Multi-Pronunciation Words:")
    multi_pron = [(w, p) for w, p in pron_dict.items() if len(p) > 1]
    for w, p in random.sample(multi_pron, min(5, len(multi_pron))):
        print(f"  {w}: {p}")

    # 5. Total word count
    print(f"\n📦 Total Words in CMU Dictionary: {len(pron_dict)}")

# -----------------------------
# 🚀 MAIN LOGIC
# -----------------------------

# Find optimal diverse word set (random trial or exhaustive)
print("🔍 Searching for the most phonetically diverse set of words...")
if USE_RANDOM_SEARCH:
    selected_words, score = find_best_set_randomized(words_by_letter, trials=TRIALS)
else:
    selected_words, score = find_best_set_exhaustive(words_by_letter)

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
    dist = phoneme_distance(w1, w2)
    G.add_edge(w1, w2, weight=dist)

# Draw graph
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, weight='weight', seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
plt.title("Phoneme Dissimilarity Graph for Selected Words")
plt.show()
