-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA journal_mode = MEMORY;
PRAGMA temp_store = MEMORY;

-- ============================================================
-- TABLE: words
-- ============================================================
-- Stores all CMU dictionary entries with rich precomputed metadata.
-- Each row represents one pronunciation of a word (so "READ" appears twice).
-- ============================================================

CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    word TEXT NOT NULL,                        -- e.g. 'BANANA'
    lemma TEXT,                                -- base form of the word
    pronunciation_index INTEGER NOT NULL DEFAULT 0, -- index for multiple pronunciations

    phoneme_list TEXT NOT NULL,                -- e.g. 'B AH0 N AE1 N AH0'
    normalized_phoneme_list TEXT NOT NULL,     -- e.g. 'B AH N AE N AH' (stress removed)

    phoneme_count INTEGER NOT NULL,            -- number of phonemes
    unique_phoneme_count INTEGER NOT NULL,     -- distinct phonemes in pronunciation
    num_syllables INTEGER NOT NULL,            -- vowel group count

    primary_stress_index INTEGER,              -- zero-based index of primary stressed vowel
    stress_pattern TEXT,                       -- e.g. '010' for 'banana'

    word_length INTEGER NOT NULL,              -- number of letters
    orth_vowel_count INTEGER NOT NULL,         -- number of vowel letters (a,e,i,o,u,y)
    orth_consonant_count INTEGER NOT NULL,     -- number of consonant letters (all other letters)
    phon_vowel_count INTEGER NOT NULL,         -- number of vowel phonemes
    phon_consonant_count INTEGER NOT NULL,     -- number of consonant phonemes

    orth_vowel_consonant_ratio REAL,           -- vowel_count / consonant_count (orthographic)
    phon_vowel_consonant_ratio REAL,           -- vowel_count / consonant_count (phonetic) 
    grapheme_to_phoneme_ratio REAL,            -- word_length / phoneme_count

    avg_vowel_height REAL,                     -- average vowel features across all vowels in pronunciation
    avg_vowel_backness REAL,                   -- average vowel backness
    avg_vowel_roundness REAL,                  -- average vowel roundness
    avg_consonant_voicing REAL,                -- average consonant features across all consonants
    avg_consonant_place REAL,                  -- average consonant place
    avg_consonant_manner REAL,                 -- average consonant manner

    sonority_profile TEXT,                     -- optional sonority sequence, e.g. '[1,3,4,5,4,1]'
    sonority_rise_count INTEGER,               -- count of sonority rises
    sonority_fall_count INTEGER,               -- count of sonority falls

    word_frequency REAL,                       -- optional (from external corpus)
    phonotactic_probability REAL,              -- optional (from phoneme-level n-gram model)

    source TEXT DEFAULT 'CMUdict-0.7b'
);