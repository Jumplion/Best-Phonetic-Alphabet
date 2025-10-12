-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA journal_mode = MEMORY;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS word_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- references to words table (store with smaller id first to enforce unordered pairs)
    word_1 TEXT NOT NULL,
    word_2 TEXT NOT NULL,

    -- Basic distances / scores
    orth_levenshtein REAL,           -- orthographic edit distance
    phon_levenshtein REAL,           -- phonetic edit distance (phoneme-level)
    feature_weighted_phon_lev REAL,  -- phonetic distance weighted by feature differences
    score REAL,                      -- combined or task-specific score

    -- Sequence-based similarities
    lcs TEXT,                     -- longest contiguous subsequence (orthographic)

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

    -- Enforce uniqueness for unordered pairs: callers should insert with smaller id first
    UNIQUE(word_1, word_2)
);
