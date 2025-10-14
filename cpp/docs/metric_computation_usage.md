# Computing and Updating Specific Metrics in average_stats

The `DBWriter` class now supports computing and updating specific metrics (like weighted phonetic Levenshtein distance) independently of the main average stats computation.

## Key Features

1. **Selective metric computation**: Compute stats for one specific metric without recalculating all metrics
2. **Update existing rows**: Uses `UPDATE` instead of `INSERT OR REPLACE` to preserve existing data
3. **Batched processing**: Process large datasets efficiently with progress tracking
4. **Thread-safe**: Uses OpenMP for parallel computation

## Database Schema Requirements

Before using these functions, ensure your `average_stats` table has columns for your metric:

```sql
ALTER TABLE average_stats ADD COLUMN avg_weighted_phon_lev REAL;
ALTER TABLE average_stats ADD COLUMN min_weighted_phon_lev REAL;
ALTER TABLE average_stats ADD COLUMN max_weighted_phon_lev REAL;
ALTER TABLE average_stats ADD COLUMN stddev_weighted_phon_lev REAL;
ALTER TABLE average_stats ADD COLUMN count_close_weighted_phon_lev INTEGER;
```

The column naming convention is: `{stat_type}_{column_prefix}` where:
- `stat_type` is one of: `avg`, `min`, `max`, `stddev`, `count_close`
- `column_prefix` is the metric identifier (e.g., `weighted_phon_lev`)

## Usage Example

### 1. Basic Usage with Weighted Phonetic Levenshtein

```cpp
#include "wordpair/db_writer.h"
#include "wordpair/scorer.h"

int main() {
    // Initialize database
    DBWriter db("data/BestPhonetics.db");
    if (!db.init()) {
        std::cerr << "Failed to initialize database\n";
        return 1;
    }

    // Preload phoneme distances (required for weighted scoring)
    if (!preload_phoneme_feature_distances("data/BestPhonetics.db")) {
        std::cerr << "Failed to preload phoneme distances\n";
        return 1;
    }

    // Load words
    std::cout << "Loading words...\n";
    auto words = db.load_all_words();
    std::cout << "Loaded " << words.size() << " words\n";

    // Compute and update weighted phonetic Levenshtein statistics
    size_t updated = db.compute_and_update_metric_batched(
        words,
        "weighted_phon_lev",                      // column prefix
        weighted_phonetic_levenshtein_score,      // scoring function
        1000,                                      // batch size
        5.0f                                       // close threshold
    );

    std::cout << "Updated " << updated << " rows\n";
    return 0;
}
```

### 2. Computing Multiple Metrics

```cpp
// Compute weighted phonetic Levenshtein
db.compute_and_update_metric_batched(
    words, 
    "weighted_phon_lev",
    weighted_phonetic_levenshtein_score,
    1000,
    5.0f
);

// Compute orthographic Jaccard index
db.compute_and_update_metric_batched(
    words,
    "orth_jaccard",
    [](const std::vector<std::string>& a, const std::vector<std::string>& b) -> float {
        // Convert phoneme vectors back to words for orthographic comparison
        // (you'd need to store/pass the actual word strings for this)
        return 0.0f;  // placeholder
    },
    1000,
    0.3f  // 30% similarity threshold
);
```

### 3. Custom Scoring Function

```cpp
// Define a custom scoring function
float custom_phoneme_score(
    const std::vector<std::string>& a, 
    const std::vector<std::string>& b
) {
    // Your custom logic here
    float score = 0.0f;
    // ... compute score ...
    return score;
}

// Use it
db.compute_and_update_metric_batched(
    words,
    "custom_metric",
    custom_phoneme_score,
    500,   // smaller batch size for complex computations
    10.0f
);
```

## Function Parameters

### `compute_and_update_metric_batched`

```cpp
size_t compute_and_update_metric_batched(
    const std::vector<Word>& words,
    const std::string& column_prefix,
    float (*score_function)(const std::vector<std::string>&, const std::vector<std::string>&),
    size_t batch_size = 1000,
    float close_threshold = 5.0f
);
```

**Parameters:**
- `words`: Vector of Word objects to process
- `column_prefix`: Database column prefix (e.g., "weighted_phon_lev")
- `score_function`: Function pointer that takes two phoneme vectors and returns a float score
- `batch_size`: Number of words to process before writing to database (default: 1000)
- `close_threshold`: Threshold for counting "close" matches (default: 5.0)

**Returns:** Number of rows successfully updated

**Scoring Function Signature:**
```cpp
float score_function(
    const std::vector<std::string>& phonemes_a,
    const std::vector<std::string>& phonemes_b
) -> float
```

## Performance Notes

1. **Batch size**: Larger batches reduce database I/O but use more memory
   - For 75K words: 500-1000 is recommended
   - Smaller batches provide more frequent progress updates

2. **Parallelization**: Uses OpenMP for parallel computation
   - Set `OMP_NUM_THREADS` environment variable to control thread count
   - Default: uses all available CPU cores

3. **Memory usage**: Each batch stores statistics for `batch_size` words
   - `WordMetricStats` is ~40 bytes per word
   - 1000 words ≈ 40KB per batch (negligible)

4. **Database locking**: Uses transactions to batch updates
   - Each batch is one transaction
   - Database is locked during transaction commits

## Error Handling

The functions handle errors gracefully:

- Returns 0 if database is not initialized
- Rolls back transaction on SQL errors
- Continues processing remaining words if individual score computation fails
- Prints detailed error messages to stderr

## Verifying Results

After computation, verify the results with SQL:

```sql
-- Check if columns were populated
SELECT 
    word_id,
    avg_weighted_phon_lev,
    min_weighted_phon_lev,
    max_weighted_phon_lev,
    stddev_weighted_phon_lev,
    count_close_weighted_phon_lev
FROM average_stats
WHERE word_id <= 10;

-- Get statistics about the metric
SELECT 
    COUNT(*) as total_words,
    AVG(avg_weighted_phon_lev) as overall_avg,
    MIN(min_weighted_phon_lev) as global_min,
    MAX(max_weighted_phon_lev) as global_max,
    AVG(stddev_weighted_phon_lev) as avg_stddev
FROM average_stats
WHERE avg_weighted_phon_lev IS NOT NULL;
```

## Integration with Existing Code

The new functions integrate seamlessly with existing workflows:

```cpp
// 1. Initial setup: compute all basic metrics
db.compute_and_write_average_stats_batched(words, 1000);

// 2. Later: add weighted phonetic Levenshtein
preload_phoneme_feature_distances("data/BestPhonetics.db");
db.compute_and_update_metric_batched(
    words,
    "weighted_phon_lev",
    weighted_phonetic_levenshtein_score,
    1000,
    5.0f
);

// 3. Filter words using both metrics
auto filtered = db.load_filtered_words(20.0, 15.0, 5);
```

## Thread Safety

- `compute_and_update_metric_batched` is thread-safe for concurrent reads
- Do not call multiple instances simultaneously on the same database
- The phoneme distance cache (`phoneme_distance_map`) is protected by a mutex
