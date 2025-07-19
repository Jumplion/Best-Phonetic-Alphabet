import os
import csv
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

""" Reads a csv file for a specific letter pair (e.g., A-B) and returns it as a dictionary. """
def read_letter_pair_scores(target_letter, compare_letter, filename_template, csv_headers):
    filename = os.path.join("CSV Files", filename_template.format(target_letter, compare_letter))
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
            for h in csv_headers:
                data[h] = row[csv_headers.index(h)]

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


def read_word_averages(filename="word_averages.csv"):
    filename = os.path.join("CSV Files", filename)

    if not os.path.exists(filename):
        logging.warning(f"Word averages file '{filename}' does not exist.")
        return None

    word_averages = defaultdict(float)
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading Word Averages", unit=" word", colour="blue"):
            word_averages[row["Word"]] = float(row["Avg_Score"])

    return word_averages


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