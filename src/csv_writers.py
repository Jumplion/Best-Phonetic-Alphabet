import os
import csv
import math
import logging
import sqlite3
import string
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from scoring import _score_candidate
from phoneme_utils import PHONEME_COORDINATES

# CSV Headers
CSV_WORD_PAIR_HEADERS:list[str] = [
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

logging.basicConfig(level=logging.INFO)

""" Creates a CSV file calculating the scores between each word (plus other stats)
Segments each Letter-Pair into their own files (e.g., A-B, A-C, etc...)
NOTE: Does not create redundant letter pairs
- E.G., A-B and B-A are not created separately since they would be identical (just reversed).
"""
def write_word_pair_scores(p_dict, p_dict_norm, words_by_letters, csv_headers=CSV_WORD_PAIR_HEADERS):

    csv_base_dir = os.path.join("data", "CSV_Files")
    os.makedirs(csv_base_dir, exist_ok=True)

    LETTERS = list(string.ascii_uppercase)  # A-Z
    filename_template = "word_pairs_{0}_{1}_data.csv"

    logging.info("Letters: " + ", ".join(LETTERS))

    # Start the main loop to create letter pairs
    for l1 in tqdm(LETTERS, total=len(LETTERS), desc="Calculating Distances...", unit=" letter", leave=False, colour="green"):

        # Get letters after l1 (including l1 itself) This avoids redundant pairs like A-B and B-A
        l1_index = LETTERS.index(l1)
        l2_letters = LETTERS[l1_index + 1:] if l1_index + 1 < len(LETTERS) else []

        for l2 in tqdm(l2_letters, total=len(l2_letters), desc=f"Calculating {l1}-Letter Pairs", unit=" letter pair", leave=False, colour="red"):  # Only create pairs (A-B, A-C, ..., B-C, ..., Z-Z)
            csv_filename = os.path.join(csv_base_dir, filename_template.format(l1, l2))    
            
            word_list1 = words_by_letters[l1]
            word_list2 = words_by_letters[l2]

            # Generate all unique word pairs (w1, w2) where w1 is from word_list1 and w2 is from word_list2
            # We make sure w1 < w2 (alphabetized) to avoid duplicate pairs like (word1, word2) and (word2, word1)
            pairs = [(w1, w2) for w1 in word_list1 for w2 in word_list2 if w1 < w2]

            if not pairs:
                logging.warning(f"No unique word pairs found for letters {l1} and {l2}. Skipping...")
                continue

            # Calculate every word pair's data and Write the results to CSV file for this letter pair
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

                # Calculate distances for each word pair
                for w1, w2 in tqdm(pairs, total=len(pairs), desc=f"Calculating and Writing to CSV: {l1}_{l2}", unit=" word pair", leave=False, colour="yellow"):
                    data = _score_candidate(selected_words=[w1, w2], phoneme_suffix_length=2, weights=None)
                    p1, p2 = p_dict[w1][0], p_dict[w2][0]
                    # Write the data to the main CSV file   
                    writer.writerow([w1, w2, data["score"],
                                    p1, p2,
                                    data["total_levenshtein"], data["total_phoneme_distance"],
                                    data["shared_sequence"][0], data["shared_sequence"][1],
                                    data["shared_suffix"][0], data["shared_suffix"][1],
                                    data["rhyme"][0], data["rhyme"][1],
                                    data["phoneme_audio_distance"]])   

def write_word_averages(p_dict, p_dict_norm, words_by_letter, csv_headers=CSV_WORD_AVERAGE_HEADERS):

    logging.info("Calculating Word Averages...")
    csv_filename = os.path.join("data", "CSV_Files", "word_averages.csv")

    # Create the CSV file and write the header. We'll write the results as we calculate
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)     
        writer.writerow(csv_headers)

    # For loops written in a way so we don't constantly go
    #   "gimmie the dictionary with all the words that don't start with this letter" for each word
    # probably a better way to do this, bleh
    LETTERS = string.ascii_uppercase  # A-Z
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
            phonemes = p_dict_norm[word][0]
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
                    result = _score_candidate([word, other_word], p_norm_dict=p_dict_norm, phoneme_suffix_length=2, weights=None)
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
                row = [word_data.get(h.lower(), "") for h in csv_headers]
                writer.writerow(row)


    logging.info(f"Word averages saved to '{csv_filename}'")
    return word_averages

def update_word_averages():

    logging.info("Calculating Word Averages...")
    # Since we are running the program in the src folder
    database_filename = os.path.join("..", "data", "phoneme_data.db")

    # For SQLite database connection
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()

    # For each row of the cmudict table — read words as plain strings (not tuples)
    cursor.execute("SELECT word FROM cmudict")
    cmu_rows = cursor.fetchall()
    cmu_words = [r[0] for r in cmu_rows]

    for word in tqdm(cmu_words, desc="Processing Words", unit="word", colour="blue", leave=False):
        running_scores: list[float] = []
        # Calculate scores against all other words starting with different letters
        for other_word in tqdm(cmu_words, desc=f"Scoring {word}", unit="comparison", leave=False, colour="red"):
            # skip comparisons with words that start with the same letter
            if not other_word or other_word[0].upper() == word[0].upper():
                continue
            result = _score_candidate([word, other_word])
            running_scores.append(float(result.get("score", 0.0)))

        avg = float(np.mean(running_scores)) if running_scores else 0.0

        # Update the cmudict table with the new average score (use parameterized query)
        cursor.execute("UPDATE cmudict SET avg_score = ? WHERE word = ?", (avg, word))
        conn.commit()

    logging.info("Word averages updated in the database.")
    conn.close()


def write_phoneme_coord_distance_dict(p_coords, filename):
    # Sort PHONEME_COORDINATES by phoneme name for consistent ordering
    distances = defaultdict(float)
    sorted_phonemes = sorted(p_coords.keys())

    for p1 in tqdm(sorted_phonemes, desc="Calculating Phoneme Distances", unit=" phoneme", colour="green"):
        coord1 = p_coords[p1]
        
        # Calculate distance from origin (0, 0, 0, 0) to each phoneme coordinate
        distances[(p1, p1)] = 0.0

        for p2 in sorted_phonemes:
            if p1 < p2:
                coord2 = p_coords[p2]
                distances[(p1, p2)] = distances[(p2, p1)] = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    # Save the phoneme distance dictionary to a file
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phoneme 1", "Phoneme 2", "Distance"])
        for (p1, p2), dist in tqdm(distances.items(), desc="Writing Phoneme Distance Dictionary", unit=" entry"):
            writer.writerow([p1, p2, dist])
    logging.info(f"Phoneme Coordinate Distance Dictionary Saved to '{filename}'")

    return distances
