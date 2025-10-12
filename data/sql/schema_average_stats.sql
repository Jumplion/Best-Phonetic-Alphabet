-- SQLite pragmas for performance and integrity

PRAGMA foreign_keys = ON;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = WAL;
PRAGMA temp_store = MEMORY;

-- ============================================================
-- average_stats table
-- ============================================================
-- Stores aggregate statistics for each word computed against all other words.
-- Used for pre-filtering: words with poor average scores (too dissimilar to all others)
-- can be excluded from the word_pairs table to save massive space.
--
-- Example: A word with avg_orth_levenshtein = 25 is very different from most words
-- and unlikely to be a good phonetic alphabet candidate.
-- ============================================================

CREATE TABLE IF NOT EXISTS average_stats (
    word_id INTEGER PRIMARY KEY,
    
    -- Average distances across all pairs involving this word
    avg_orth_levenshtein REAL NOT NULL,     -- avg orthographic distance vs all other words
    avg_phon_levenshtein REAL NOT NULL,     -- avg phonetic distance vs all other words
    avg_lcs_length REAL NOT NULL,           -- avg longest common subsequence length
    
    -- Distribution statistics (optional but useful for filtering)
    min_orth_levenshtein INTEGER,           -- closest orthographic match
    max_orth_levenshtein INTEGER,           -- furthest orthographic match
    stddev_orth_levenshtein REAL,           -- standard deviation (measures consistency)
    
    min_phon_levenshtein INTEGER,           -- closest phonetic match
    max_phon_levenshtein INTEGER,           -- furthest phonetic match
    stddev_phon_levenshtein REAL,
    
    -- Count of "good matches" (words within threshold)
    count_close_orth INTEGER,               -- count where orth_lev <= 10
    count_close_phon INTEGER,               -- count where phon_lev <= 10
    
    -- Metadata
    computed_against_n_words INTEGER,       -- how many words were compared
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY(word_id) REFERENCES words(id) ON DELETE CASCADE
);

-- Index for filtering queries
CREATE INDEX IF NOT EXISTS idx_avg_stats_orth ON average_stats(avg_orth_levenshtein);
CREATE INDEX IF NOT EXISTS idx_avg_stats_phon ON average_stats(avg_phon_levenshtein);
CREATE INDEX IF NOT EXISTS idx_avg_stats_close_count ON average_stats(count_close_orth, count_close_phon);

-- ============================================================
-- Filtering Strategy Examples
-- ============================================================
-- After populating average_stats, use it to filter words before computing pairs:
--
-- Example 1: Only process words with good average scores
-- SELECT word_id FROM average_stats 
-- WHERE avg_orth_levenshtein <= 15 AND avg_phon_levenshtein <= 12;
--
-- Example 2: Only words with at least some close matches
-- SELECT word_id FROM average_stats 
-- WHERE count_close_orth >= 10 AND count_close_phon >= 10;
--
-- Example 3: Words in the "sweet spot" (not too similar, not too different)
-- SELECT word_id FROM average_stats 
-- WHERE avg_orth_levenshtein BETWEEN 8 AND 18 
--   AND avg_phon_levenshtein BETWEEN 6 AND 15;
--
-- This filtering can eliminate 50-80% of words from pair computation,
-- reducing the total pairs from 2.83B to 100-700M (90-97% reduction!)
-- ============================================================
