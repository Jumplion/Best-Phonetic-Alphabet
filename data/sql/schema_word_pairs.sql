-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA temp_store = MEMORY;

-- ============================================================
-- word_pairs table
-- ============================================================
CREATE TABLE IF NOT EXISTS word_pairs (
    -- Use integer IDs instead of text (references words.id)
    -- Ensures smaller id is first to enforce unordered pairs
    word_id_1 INTEGER NOT NULL,
    word_id_2 INTEGER NOT NULL,

    -- Basic distances (INTEGER type optimizes to 1-2 bytes for small values)
    orth_levenshtein small INTEGER NOT NULL,      -- orthographic edit distance (0-255 typical)
    phon_levenshtein small INTEGER NOT NULL,      -- phonetic edit distance (0-255 typical)

    -- LCS length only (not the full text - can regenerate if needed)
    lcs_length small INTEGER NOT NULL,            -- longest contiguous subsequence length

    -- Primary key on the pair
    -- Composite key acts as the table's storage key due to WITHOUT ROWID
    PRIMARY KEY (word_id_1, word_id_2),
    
    -- Foreign key constraints to words table
    FOREIGN KEY(word_id_1) REFERENCES words(id) ON DELETE CASCADE,
    FOREIGN KEY(word_id_2) REFERENCES words(id) ON DELETE CASCADE
) WITHOUT ROWID;  -- Further optimization: eliminates rowid overhead

-- Optional: Index for reverse lookups (uncomment if needed)
-- CREATE INDEX IF NOT EXISTS idx_word_pairs_reverse ON word_pairs(word_id_2, word_id_1);

-- ============================================================
-- FUTURE COLUMNS (to be added when computed):
-- ============================================================
-- When adding new metrics, consider using ALTER TABLE to add columns
-- rather than recreating the table. Examples:
--
-- ALTER TABLE word_pairs ADD COLUMN feature_weighted_phon_lev REAL;
-- ALTER TABLE word_pairs ADD COLUMN prefix_orth_overlap REAL;
-- ALTER TABLE word_pairs ADD COLUMN phoneme_bigram_jaccard REAL;
-- etc.
-- ============================================================
