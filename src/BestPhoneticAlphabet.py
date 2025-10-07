import os
import csv
import logging
import json
import string
import time
from collections import defaultdict

# Scientific and Data Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import cmudict, wordnet
from nltk.stem import WordNetLemmatizer

import sqlite3
from datetime import datetime

from scoring import (candidate_gen, normalize_phoneme, _score_candidate)

# Constants and Settings
LETTERS = list(string.ascii_uppercase)

# Phoneme Coordinates
PHONEME_DICT = defaultdict(list)
WORDS_BY_LETTER = defaultdict(list)

# Word and Phoneme Blacklists and Whitelists
# TODO: Convert to JSON User Settings
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

# TODO: Convert to configurable user settings in the JSON
""" Letter Prefixes that should not be at the start of words
These prefixes are often silent or not pronounced, so we filter them out
"""
BANNED_LETTER_PREFIXES = [
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

# TODO: Convert to configurable user settings in the JSON
""" Phoneme Prefixes that are banned from certain letter groups.
For example, "E" words shouldn't start with "y" like "eunuch" or "euphoria"
"""
BANNED_PHONEME_PREFIXES = {
    "a" : [],
    "b" : [],
    "c" : ["S"],        # "cereal", "cell"
    "d" : [],
    "e" : ["ER", "Y"],  # "ernest" | "eunuch", "euphoria"
    "f" : [],
    "g" : ["N"],        # "gnome"
    "h" : ["AW", "OW"], # "hour"
    "i" : [],
    "j" : [],
    "k" : ["N"],        # "knight"
    "l" : [],
    "m" : ["N"],        # "mnemonic"
    "n" : [],
    "o" : [],
    "p" : ["F"],        # "phone", "philosophy"
    "q" : [],
    "r" : [],
    "s" : ["SH"],       # "she", "sure"
    "t" : ["TH", "DH"], # "this" | "that"
    "u" : [],
    "v" : [],    
    "w" : ["R", "HH"],  # 'wrestle' | 'whole'
    "x" : [],
    "y" : [],
    "z" : []
}

# -------------------------------
# Dictionary Functions
# -------------------------------

""" Clean the CMU Pronouncing Dictionary and apply filters.
- Filters out words with non-alphabetic characters, too short/long words, blacklisted words, etc.
- Lemmatizes words to their base forms and groups them by lemma, selecting the shortest variant.
- Adds custom words with predefined phonemes.
"""
def get_cleaned_cmu_dict():
    logging.info("Cleaning CMU Dictionary...")

    # Get user settings
    settings = load_user_settings()
    
    # Get Settings
    min_phoneme_length = settings["min_phoneme_length"]
    max_phoneme_length = settings["max_phoneme_length"]
    min_word_length = settings["min_word_length"]
    max_word_length = settings["max_word_length"]
    min_syllables = settings["min_syllables"]
    max_syllables = settings["max_syllables"]

    # Get Filters
    filter_phoneme_length = settings["filter_phoneme_length"]
    filter_word_length = settings["filter_word_length"]
    filter_syllables = settings["filter_syllables"]
    filter_letter_prefixes = settings["filter_letter_prefixes"]
    filter_phoneme_prefixes = settings["filter_phoneme_prefixes"]
    filter_averages = settings["filter_averages"]

    word_averages = None #read_word_averages()
    if word_averages is None:
        logging.error("Word averages file not found. Dictionary won't filter out words based on averages.")
    min_avg = np.mean(list(word_averages.values())) if word_averages else 0.0

    cleaned_dict = cmudict.dict()

    cleaned_dict = {
        w: prons for w, prons in tqdm(cleaned_dict.items(), desc="Filtering Words from CMU Dictionary", unit=" word")
        if (
            (w in WORD_WHITELIST) or
            (
                # TODO: Redo this filtering system because if any of the filters are false, 
                # then the word is not added instead of that filter being ignored, D'OH
                (w not in WORD_BLACKLIST)
                and (w[0].isalpha())
                and (wordnet.synsets(w))
                and (filter_phoneme_length      and (min_phoneme_length <= len(prons[0]) <= max_phoneme_length))
                and (filter_word_length         and (min_word_length <= len(w) <= max_word_length))
                and (filter_syllables           and (min_syllables <= len([p for p in prons[0] if p[-1].isdigit()]) <= max_syllables))
                and (filter_letter_prefixes     and not any(w.startswith(prefix) for prefix in BANNED_LETTER_PREFIXES))
                and (filter_phoneme_prefixes    and not any(w.startswith(prefix) for prefix in BANNED_PHONEME_PREFIXES.get(w[0].lower(), [])))
                and (filter_averages            and (word_averages[w] >= min_avg if word_averages else True))
            )
        )
    }              

    lemmatizer = WordNetLemmatizer()
    lemma_map = defaultdict(list)
    
    # Lemmatize the words, group by lemma (in cleaned dictionary)
    for w, prons in tqdm(cleaned_dict.items(), desc="Lemmatizing...", unit=" word"):
        noun = lemmatizer.lemmatize(w.lower(), pos='n')
        lemma = lemmatizer.lemmatize(noun, pos='v')
        lemma_map[lemma].append((w, prons[0]))

    # Choose representative word for each lemma (shortest spelling)
    lemma_dict = {}
    for lemma, variants in lemma_map.items():
        representative = min(variants, key=lambda x: len(x[0]))
        lemma_dict[representative[0]] = [representative[1]]  # keep as list for CMU compatibility 
    
    cleaned_dict = lemma_dict

    # Add Custom Words to the dictionary
    for w, pron in CUSTOM_WORDS.items():
        cleaned_dict[w.lower()] = [pron]

    logging.info(f"CMU Pronouncing Dictionary Cleaned | Total Words: {len(cleaned_dict)}")
    return cleaned_dict

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def load_user_settings(settings_path="../user_settings.json"):
    """ Load user settings from a JSON file. """
    filepath = os.path.join(os.path.dirname(__file__), settings_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Settings file not found: {filepath}")
    with open(filepath, "r") as f:
        settings = json.load(f)
    return settings

# -------------------------------
# Main Function
# -------------------------------

def main():   
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detail, WARNING for less
        format='[%(levelname)s]: %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            # Uncomment below to also log to a file:
            # logging.FileHandler("phonetic_alphabet.log")
        ]
    )
    logging.info("Starting Best Phonetic Alphabet Utility")
    
    logging.info("Downloading CMU Dictionary")
    nltk.download('cmudict')
    nltk.download('wordnet')

    # TODO: Rewrite so that it utilizes the phoneme_data.db stuff
    PHONEME_DICT = None #get_cleaned_cmu_dict()


    choices = {
        '1': "Find Best (Randomized Trial)",
        '2': "Score Premade Alphabet"
    }

    logging.info("Best Phonetic Alphabet Utility")
    print("Choose an option:")
    for key, value in choices.items():
        print(f"{key}. {value}")
    
    user_input = input(f"Enter your choice ({choices.keys()}): ").strip().lower()
    
    choice = choices[user_input] if user_input in choices else None
    
    if not choice:
        print("Invalid choice. Please run the program again.")
        return

    else:
        print("Exiting Program.")

if __name__ == "__main__":
    main()