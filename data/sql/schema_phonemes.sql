-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA journal_mode = MEMORY;
PRAGMA temp_store = MEMORY;

-- ============================================================
-- TABLE: phonemes
-- ============================================================
-- One row per ARPAbet phoneme.
-- Each phoneme is annotated with its articulatory features,
-- stored both as individual numeric columns and as a JSON vector.
-- ============================================================

CREATE TABLE IF NOT EXISTS phonemes (
    phoneme TEXT PRIMARY KEY,
    is_vowel INTEGER NOT NULL,

    -- vowel columns (nullable for consonants)
    height REAL,        --  0 = Low [Open], 0.5 = Mid, 1 = High [Close]
    backness REAL,      --  0 = Front,  0.5 = Central, 1 = Back
    roundness REAL,     --  0 = Rounded, 1 = Unrounded
    
    -- consonant columns (nullable for vowels)
    place REAL,         -- 0 = Bilabial, 0.2 = Labiodental, 0.4 = Dental, 0.6 = Alveolar, 0.8 = Velar, 1 = Glottal
    manner REAL,        -- 0 = Stop, 0.125 = Affricate, 0.25 = Fricative, 0.5 = Nasal, 0.75 = Lateral Liquid, 0.875 = Rhotic Liquid, 1 = Glide
    voicing REAL,       -- 0 = Voiceless, 1 = Voiced

    feature_vector JSON    -- e.g. '[1, 0.5, 0, 0.75, 0.8, 1]'
);

CREATE TABLE IF NOT EXISTS phoneme_distances (
    phoneme_1 TEXT NOT NULL,
    phoneme_2 TEXT NOT NULL,
    feature_based_distance REAL NOT NULL,
    audio_based_distance REAL NOT NULL,
    PRIMARY KEY (phoneme_1, phoneme_2)
);