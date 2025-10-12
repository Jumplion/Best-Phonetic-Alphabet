-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA journal_mode = MEMORY;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS word_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- references to words table (store with smaller id first to enforce unordered pairs)
    word_id_1 INTEGER NOT NULL,
    word_id_2 INTEGER NOT NULL,

    -- Basic distances / scores
    orth_levenshtein REAL,           -- orthographic edit distance
    phon_levenshtein REAL,           -- phonetic edit distance (phoneme-level)
    feature_weighted_phon_lev REAL,  -- phonetic distance weighted by feature differences
    score REAL,                      -- combined or task-specific score

    -- Sequence-based similarities
    lcs INTEGER,                     -- longest common subsequence length (orthographic)

    -- Prefix / suffix overlap (normalized ratios)
    prefix_orth_overlap REAL,
    suffix_orth_overlap REAL,
    prefix_phon_overlap REAL,
    suffix_phon_overlap REAL,

    -- n-gram overlap metrics (scalar) and flexible JSON for detailed counts
    phoneme_bigram_jaccard REAL,
    phoneme_trigram_jaccard REAL,
    phoneme_ngram_overlap JSON,

    -- Rhyme / prosody features
    rhyme_similarity REAL,
    stress_pattern_similarity REAL,
    syllabic_structure_match INTEGER,

    -- Path / transition and neighborhood features
    phonetic_path_smoothness REAL,
    neighborhood_density_diff REAL,

    -- Generic set similarity (can be used for grapheme/phoneme sets)
    jaccard_index REAL,

    -- bookkeeping
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_word1 FOREIGN KEY(word_id_1) REFERENCES words(id) ON DELETE CASCADE,
    CONSTRAINT fk_word2 FOREIGN KEY(word_id_2) REFERENCES words(id) ON DELETE CASCADE,

    -- Enforce uniqueness for unordered pairs: callers should insert with smaller id first
    UNIQUE(word_id_1, word_id_2)
);
