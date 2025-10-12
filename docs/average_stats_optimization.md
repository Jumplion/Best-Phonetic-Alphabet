# Average Statistics Pre-Filtering Optimization

## Overview

This optimization dramatically reduces the computational cost of finding the best phonetic alphabet by pre-computing aggregate statistics for each word and using those to filter out poor candidates before computing word pairs.

## How It Works

### 1. Compute Average Statistics

For each word in the database, we compute its average distances against all other words:

- **avg_orth_levenshtein**: Average orthographic Levenshtein distance
- **avg_phon_levenshtein**: Average phonetic Levenshtein distance  
- **avg_lcs_length**: Average longest common subsequence length
- **count_close_orth**: Number of words within orthographic threshold (≤5)
- **count_close_phon**: Number of words within phonetic threshold (≤4)
- **min/max distances**: Range of distances encountered

### 2. Filter by Average Quality

Words with poor average scores are unlikely to be good phonetic alphabet candidates. By setting thresholds on average statistics, we can eliminate 50-90% of words before computing pairs:

```sql
SELECT w.id, w.word, w.normalized_phoneme_list
FROM words w
INNER JOIN average_stats a ON w.id = a.word_id
WHERE a.avg_orth_levenshtein <= 6.0
  AND a.avg_phon_levenshtein <= 5.5
  AND a.count_close_phon >= 10
```

### 3. Compute Pairs Only for Filtered Words

After filtering, we only compute word pairs for the remaining "good candidate" words, dramatically reducing the total number of pairs.

## Expected Savings

Based on testing with 1,000 words from the dataset:

| Filter Threshold | Words Kept | % Reduction | Pairs Before | Pairs After | Storage Saved |
|-----------------|------------|-------------|--------------|-------------|---------------|
| avg_orth ≤ 5.5, avg_phon ≤ 5.0 | 16 | 98.4% | 499,500 | 120 | 99.98% |
| avg_orth ≤ 6.0, avg_phon ≤ 5.5 | 326 | 67.4% | 499,500 | 52,975 | 89.4% |
| avg_orth ≤ 6.5, avg_phon ≤ 6.0 | 600 | 40.0% | 499,500 | 179,700 | 64.0% |

Extrapolating to full dataset (75,267 words):

| Filter | Words Kept | Pairs | Database Size | Time to Compute |
|--------|------------|-------|---------------|-----------------|
| No filter | 75,267 | 2.83B | 104 GB | ~300+ hours |
| Moderate (6.0/5.5) | ~24,000 | 288M | 10.5 GB | ~10 hours |
| Strict (5.5/5.0) | ~1,200 | 720K | 26 MB | ~20 minutes |

## Usage

### Step 1: Compute Average Statistics

This is a one-time computation that takes ~2-3 hours for the full dataset:

```bash
# Compile with OpenMP support
cd "f:/Repos/Best Phonetic Alphabet"
cmake --build build --config Release

# Run computation for all words (no limit)
./build/src/test_average_stats.exe 75267

# Or test with subset first
./build/src/test_average_stats.exe 5000
```

### Step 2: Analyze Statistics to Choose Thresholds

Query the database to understand the distribution:

```sql
-- Overall statistics
SELECT 
    COUNT(*) as total_words,
    MIN(avg_orth_levenshtein) as min_avg_orth,
    MAX(avg_orth_levenshtein) as max_avg_orth,
    AVG(avg_orth_levenshtein) as overall_avg_orth,
    MIN(avg_phon_levenshtein) as min_avg_phon,
    MAX(avg_phon_levenshtein) as max_avg_phon,
    AVG(avg_phon_levenshtein) as overall_avg_phon
FROM average_stats;

-- Count words at different thresholds
SELECT 
    COUNT(*) FILTER (WHERE avg_orth_levenshtein <= 5.5 AND avg_phon_levenshtein <= 5.0) as very_strict,
    COUNT(*) FILTER (WHERE avg_orth_levenshtein <= 6.0 AND avg_phon_levenshtein <= 5.5) as strict,
    COUNT(*) FILTER (WHERE avg_orth_levenshtein <= 6.5 AND avg_phon_levenshtein <= 6.0) as moderate,
    COUNT(*) FILTER (WHERE avg_orth_levenshtein <= 7.0 AND avg_phon_levenshtein <= 6.5) as loose,
    COUNT(*) as total
FROM average_stats;

-- Find words with best average scores
SELECT w.word, a.avg_orth_levenshtein, a.avg_phon_levenshtein, 
       a.count_close_orth, a.count_close_phon
FROM words w
JOIN average_stats a ON w.id = a.word_id
ORDER BY a.avg_phon_levenshtein ASC, a.avg_orth_levenshtein ASC
LIMIT 50;
```

### Step 3: Modify main.cpp to Use Filtering

Add command-line arguments to the main scoring program:

```cpp
// Add to main.cpp argument parsing:
double avg_orth_threshold = 6.0;
double avg_phon_threshold = 5.5;
int min_close_matches = 10;

for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--avg-orth-threshold" && i + 1 < argc) {
        avg_orth_threshold = std::atof(argv[++i]);
    } else if (arg == "--avg-phon-threshold" && i + 1 < argc) {
        avg_phon_threshold = std::atof(argv[++i]);
    } else if (arg == "--min-close-matches" && i + 1 < argc) {
        min_close_matches = std::atoi(argv[++i]);
    }
}

// Replace load_all_words() with:
std::vector<Word> words = db_writer.load_filtered_words(
    avg_orth_threshold,
    avg_phon_threshold,
    min_close_matches
);
```

### Step 4: Run Full Computation with Filtering

```bash
# With moderate filtering (recommended)
./build/src/wordpair_scorer.exe ^
    --avg-orth-threshold 6.0 ^
    --avg-phon-threshold 5.5 ^
    --min-close-matches 10 ^
    --batch-size 100000

# With strict filtering (fastest, smallest output)
./build/src/wordpair_scorer.exe ^
    --avg-orth-threshold 5.5 ^
    --avg-phon-threshold 5.0 ^
    --min-close-matches 20 ^
    --batch-size 100000
```

## Performance Characteristics

### Computation Time

- **Average stats computation**: O(n²) where n = total words
  - Single-threaded: ~10 seconds per 1,000 words
  - 16-thread parallel: ~1.5 seconds per 1,000 words
  - Full dataset (75K words): ~2.4 hours with parallelization

- **Pair scoring with filtering**: O(m²) where m = filtered words
  - With 67% filtering: 288M pairs instead of 2.83B (10x reduction)
  - Time savings: ~90% reduction in computation time

### Storage Requirements

- **average_stats table**: ~150 bytes per word × 75K words ≈ 11 MB
- **word_pairs table**: 40.5 bytes per pair
  - No filtering: 2.83B pairs = 104 GB
  - With filtering: 288M pairs = 10.5 GB (90% savings)

### Memory Requirements

- Average stats computation: negligible (streaming results)
- Filtered pair computation: same as before (~1-2 GB batch buffer)

## Iteration Strategy

Since computing average stats is relatively fast (~2-3 hours), you can iterate on thresholds:

1. **First pass**: Compute average stats for all 75K words (one-time cost)
2. **Analyze**: Query distribution and identify optimal thresholds
3. **Test**: Run with strict filtering (5.5/5.0) to get quick results
4. **Iterate**: If results look good, keep strict; otherwise try moderate (6.0/5.5)
5. **Refine**: Use actual pair results to inform whether to widen or narrow filter

## Integration with Existing Code

The implementation is non-invasive:

- ✅ Existing `load_all_words()` still works (no filtering)
- ✅ New `load_filtered_words()` method for optimization
- ✅ `compute_average_stats()` is static utility (no instance needed)
- ✅ All existing code continues to work unchanged

## Next Steps

1. Run `test_average_stats.exe 75267` to compute stats for full dataset
2. Query database to analyze distribution
3. Choose initial thresholds (recommend starting with 6.0/5.5)
4. Modify main.cpp to add command-line arguments for thresholds
5. Run full computation with filtering
6. Analyze results and adjust thresholds if needed

## Technical Notes

- Average stats are computed with **OpenMP parallelization** for speed
- Database writes use **transactions** for efficiency
- Filtering uses **JOIN** with indexed columns for fast queries
- The `average_stats` table has **FOREIGN KEY** constraint for data integrity
- Progress reporting every 1,000 words during computation
