import nltk
import networkx as nx
import matplotlib.pyplot as plt
import editdistance
from collections import defaultdict
from itertools import combinations
import random

# 🔧 CONFIGURATION
FILTER_WORDS = True           # Set to True to filter out words with fewer than MIN_PHONEME_LENGTH phonemes
MAX_WORDS_PER_LETTER = 1000     # Max candidate words per starting letter
TRIALS = 2000                  # Number of random trials
MIN_PHONEME_LENGTH = 2         # Minimum number of phonemes required per word
USE_RANDOM_SEARCH = True  # Set to True to use random sampling, False to use greedy exhaustive search

# -----------------------------
# 📦 FUNCTION DEFINITIONS
# -----------------------------

def phoneme_distance(w1, w2):
    """Calculate Levenshtein distance between two words' primary phoneme lists."""
    p1 = pron_dict[w1][0]
    p2 = pron_dict[w2][0]
    return editdistance.eval(p1, p2)

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
        if len(candidate) < 26:
            continue
        score = total_distance(candidate)
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


# -----------------------------
# 📥 DATA LOADING & PREP
# -----------------------------

# Download CMU Pronouncing Dictionary if needed
nltk.download('cmudict')
from nltk.corpus import cmudict
pron_dict = cmudict.dict()

# Group words by first letter
words_by_letter = defaultdict(list)
for word in pron_dict:
    if word[0].isalpha() and word.isalpha():
        first_letter = word[0].upper()
        words_by_letter[first_letter].append(word)

# Filter and trim word lists
if FILTER_WORDS:
    for letter in words_by_letter:
        filtered = [w for w in words_by_letter[letter] if len(pron_dict[w][0]) >= MIN_PHONEME_LENGTH]
        words_by_letter[letter] = filtered[:MAX_WORDS_PER_LETTER]

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
