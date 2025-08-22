import os
import sqlite3
import string
import random
import numpy as np
import editdistance

from phoneme_utils import PHONEME_COORDINATES, PHONEME_DISTANCES 

""" Extract the rhyme portion of a phoneme list.
NOTE: Requires the phoneme list to have stress markers (e.g., '1' for primary stress).
Probably doesn't really work, doh well
"""
# NOTE: Probably doesn't work as intended, doh
def extract_rhyme_portion(phonemes):
    rhyme = []
    found_primary_stress = False
    for ph in phonemes:
        if found_primary_stress:
            rhyme.append(ph)
        elif ph[-1] == '1':  # look for primary stress
            found_primary_stress = True
            rhyme.append(ph)
    return tuple(rhyme) if found_primary_stress else tuple()

""" Remove stress digits from phonemes. """
def normalize_phoneme(p_list):
    return [p[:-1] if p[-1].isdigit() else p for p in p_list]

def candidate_gen(trials, words_by_letter, preselected_by_letter=None):
    LETTERS = string.ascii_uppercase  # A-Z
    for _ in range(trials):
        candidate = []
        for le in LETTERS:
            if le in preselected_by_letter:
                candidate.append(preselected_by_letter[le])
            elif words_by_letter[le]:
                candidate.append(random.choice(words_by_letter[le]))
        if len(candidate) == len(LETTERS):
            yield candidate

def analyze_alphabet(data):
    selected_words = data.get("selected_words", [])
    phoneme_suffix_length = data.get("phoneme_suffix_length", 2)
    weights = data.get("weights", None)

    # Score the candidate alphabet
    return _score_candidate(selected_words, phoneme_suffix_length, weights)

def _score_candidate(selected_words, phoneme_suffix_length=2, weights=None):

    # Connect to database
    db_path = os.path.join("..", "data", "phoneme_data.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get word data from database "cmudict"
    word_data = {}
    for word in selected_words:
        cur.execute("""
            SELECT word, pronunciation, norm_pron, syllables, lemma, avg_score
            FROM cmudict 
            WHERE word = ?
        """, (word,))

        result = cur.fetchone()
        word_data[word] = {
            'pronunciation': result[1].split() if result[1] else [],
            'norm_pron': result[2].split() if result[2] else [],
        }
    
    conn.close()

    total_levenshtein, total_phoneme_coord_dist, total_phoneme_audio_dist = 0, 0, 0
    shared_sequence_penalty, shared_suffix_penalty = 0, 0
    rhyme_penalty = 0

    vowels = [v for v in PHONEME_COORDINATES if PHONEME_COORDINATES[v][0] == 0]
    consonants = [c for c in PHONEME_COORDINATES if PHONEME_COORDINATES[c][0] == 1]

    vowel_set, consonant_set = set(), set()
    suffix_counts, suffix_to_words = {}, {}
    phoneme_suffix_counts, phoneme_suffix_to_words = {}, {}
    shared_sequences_list, shared_suffixes_list = [], []
    rhyme_pairs = []

    # Pairwise comparisons
    for i in range(len(selected_words)):
        w1 = selected_words[i]
        p1 = word_data[w1]['norm_pron']

        # Collect vowels and consonants
        for p in p1:
            if p in vowels:
                vowel_set.add(p)
            if p in consonants:
                consonant_set.add(p)

        # Collect orthographic suffixes of the current word (2-5 characters)
        word_len = len(w1)
        for slen in range(2, min(6, word_len + 1)):
            suffix = w1[-slen:]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            suffix_to_words.setdefault(suffix, []).append(w1)

        # Collect phoneme suffixes
        if len(p1) >= phoneme_suffix_length:
            phon_suffix = tuple(p1[-phoneme_suffix_length:])
            phoneme_suffix_counts[phon_suffix] = phoneme_suffix_counts.get(phon_suffix, 0) + 1
            phoneme_suffix_to_words.setdefault(phon_suffix, []).append(w1)
        
        # Compare to all subsequent words
        for j in range(i + 1, len(selected_words)):
            w2 = selected_words[j]
            p2 = word_data[w2]['norm_pron']

            total_levenshtein += editdistance.eval(w1, w2)

            for x in range(min(len(p1), len(p2))):
                total_phoneme_coord_dist += np.sum(PHONEME_COORDINATES.get((p1[x], p2[x]), {'coord': 0.0})['coord'])
                total_phoneme_audio_dist += np.sum(PHONEME_DISTANCES.get((p1[x], p2[x]), {'audio': 0.0})['audio'])

            # Add excess phoneme values if needed (Arbitrary penalty for excess)
            excess = abs(len(p1) - len(p2))
            total_phoneme_audio_dist += excess * 0.5
            total_phoneme_coord_dist += excess * 0.5

            # Shared phoneme sub-sequences (n-grams)
            ngram_max = min(len(p1), len(p2))
            for n in range(1, ngram_max + 1):
                ngrams1 = {tuple(p1[k:k+n]) for k in range(len(p1) - n + 1)}
                ngrams2 = {tuple(p2[k:k+n]) for k in range(len(p2) - n + 1)}
                shared_ngrams = ngrams1.intersection(ngrams2)
                for ngram in shared_ngrams:
                    shared_sequence_penalty += 1
                    shared_sequences_list.append((w1, w2, ngram))

    # Orthographic suffixes
    for suffix, count in suffix_counts.items():
        if count > 1:
            shared_suffix_penalty += (count - 1)
            shared_suffixes_list.append((f"orthographic: {suffix}", suffix_to_words[suffix]))

    # Phoneme suffixes
    for suffix, count in phoneme_suffix_counts.items():
        if count > 1:
            shared_suffix_penalty += (count - 1)
            shared_suffixes_list.append((f"phonemic: {suffix}", phoneme_suffix_to_words[suffix]))

    # Weights
    WEIGHT_LEVENSHTEIN =            weights["weight_levenshtein"] if weights is not None           else 1.0
    WEIGHT_PHONEME =                weights["weight_phoneme"] if weights is not None               else 1.0
    WEIGHT_SHARED_SEQ =             weights["weight_shared_seq"] if weights is not None            else 2.0
    WEIGHT_SHARED_SUFFIX =          weights["weight_shared_suffix"] if weights is not None         else 2.0
    WEIGHT_RHYME =                  weights["weight_rhyme"] if weights is not None                 else 2.0
    #WEIGHT_VOWEL_DIVERSITY =        weights["weight_vowel_diversity"] if weights is not None       else 1.0
    #WEIGHT_CONSONANT_DIVERSITY =    weights["weight_consonant_diversity"] if weights is not None   else 1.0
    WEIGHT_AUDIO_DIVERSITY =        weights["weight_audio_diversity"] if weights is not None       else 1.0

    score = (
        (WEIGHT_LEVENSHTEIN     * total_levenshtein)
        + (WEIGHT_PHONEME       * total_phoneme_coord_dist)
        + (WEIGHT_AUDIO_DIVERSITY * total_phoneme_audio_dist)
        # + (WEIGHT_VOWEL_DIVERSITY       * (len(vowel_set) / len(vowels)))
        # + (WEIGHT_CONSONANT_DIVERSITY   * (len(consonant_set) / len(consonants)))
        - (WEIGHT_SHARED_SEQ    * shared_sequence_penalty)
        - (WEIGHT_SHARED_SUFFIX * shared_suffix_penalty)
        - (WEIGHT_RHYME         * rhyme_penalty)
    )

    # Individual Candidate Scores totaled
    return {
        "score": score,
        "total_levenshtein": total_levenshtein,
        "total_phoneme_distance": total_phoneme_coord_dist,
        "phoneme_audio_distance": total_phoneme_audio_dist,
        "shared_sequence": (shared_sequence_penalty, shared_sequences_list),
        "shared_suffix": (shared_suffix_penalty, shared_suffixes_list),
        "rhyme": (rhyme_penalty, rhyme_pairs),
        "vowel_diversity": (len(vowel_set) / len(vowels)),
    }