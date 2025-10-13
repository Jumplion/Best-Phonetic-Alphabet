# Batched Average Statistics Computation

## Overview

The batched computation method processes words in chunks and writes results to the database incrementally, preventing memory overflow and providing resilience against crashes.

## Key Features

### 1. **Batch Processing**
- Processes words in configurable batches (default: 1000 words)
- Computes stats for batch, then writes to database immediately
- Frees memory after each batch write

### 2. **Memory Efficiency**
- Never holds more than one batch worth of stats in memory
- For 75K words with batch size 1000: only ~150KB in memory at once vs ~11MB for all

### 3. **Progress Tracking**
- Single-line progress updates every 100 words
- Shows: current progress, percentage, speed, and ETA
- Updates continuously across batches

### 4. **Crash Resilience**
- If computation is interrupted, already-written batches are saved
- Can resume by checking which words already have stats in database
- Use `INSERT OR REPLACE` to safely re-run batches

### 5. **Performance**
- Same parallel computation as non-batched version
- Only ~5-10ms overhead per batch for database write
- Total time: ~2-3 hours for 75K words (same as before)

## Usage

### Run Full Dataset

```bash
cd "f:\Repos\Best Phonetic Alphabet\build\src"
.\test_average_stats.exe
```

This will:
1. Load all 75,267 words from database
2. Process in batches of 1000 words
3. Write each batch immediately after computation
4. Show continuous progress with ETA
5. Display final summary

### Run Test Subset

```bash
# Test with 1500 words (3 batches of 500)
.\test_average_stats.exe 1500

# Test with 5000 words (5 batches of 1000)
.\test_average_stats.exe 5000
```

### Custom Batch Size (in code)

Modify `test_average_stats.cpp`:

```cpp
// Larger batches = fewer DB writes, more memory
size_t batch_size = 2000;

// Smaller batches = more DB writes, less memory
size_t batch_size = 500;
```

## Output Example

```
Computing and writing average statistics in batches...
Total words: 75267 | Batch size: 1000
Progress updates every 100 words...

--- Batch: words 0 to 999 (1000 words) ---
  Progress: 1000 / 75267 (1.33%) | Speed: 125.3 words/sec | ETA: 2h 34m 12s
  Batch computation took 8.2 seconds
  Writing batch to database...
  ✅ Wrote 1000 rows in 12 ms

--- Batch: words 1000 to 1999 (1000 words) ---
  Progress: 2000 / 75267 (2.66%) | Speed: 124.8 words/sec | ETA: 2h 31m 45s
  ...
```

## Advantages Over Non-Batched

| Feature | Non-Batched | Batched |
|---------|-------------|---------|
| Memory Usage | ~11 MB | ~150 KB per batch |
| Crash Recovery | Lose everything | Keep completed batches |
| Progress Visibility | End only | Real-time with writes |
| Database Writes | 1 large transaction | Many small transactions |
| Resumability | No | Yes (with minor code changes) |

## Performance Characteristics

### Time Breakdown (for 75K words)

- **Computation per word**: ~1.5 seconds per 100 words
- **Database write per batch**: 5-15ms per 1000 words
- **Total overhead**: <1% of total time
- **Expected total time**: 2-3 hours

### Batch Size Impact

| Batch Size | Memory | DB Writes | Write Overhead |
|------------|--------|-----------|----------------|
| 500 | ~75 KB | 151 | 1.5 seconds |
| 1000 | ~150 KB | 76 | 0.9 seconds |
| 2000 | ~300 KB | 38 | 0.5 seconds |
| 5000 | ~750 KB | 16 | 0.2 seconds |

**Recommendation**: Use 1000 (default) for good balance.

## Resuming After Interruption

If the process is interrupted, you can resume by:

1. Query which words already have stats:
```sql
SELECT word_id FROM average_stats;
```

2. Modify code to skip already-processed words:
```cpp
// Load existing word IDs
std::set<int> processed_ids = load_processed_word_ids(db_writer);

// Filter out already-processed words
std::vector<Word> remaining_words;
for (const auto& word : all_words) {
    if (processed_ids.find(word.id) == processed_ids.end()) {
        remaining_words.push_back(word);
    }
}

// Process only remaining words
db_writer.compute_and_write_average_stats_batched(remaining_words, 1000, 5, 4);
```

## Database Impact

- Uses `INSERT OR REPLACE` - safe to re-run without duplicates
- Each batch uses a transaction for efficiency
- Indexes automatically updated after each batch
- No table locks between batches

## Next Steps After Completion

Once all average stats are computed:

1. **Analyze distribution**:
```sql
SELECT 
    MIN(avg_phon_levenshtein) as min_avg,
    MAX(avg_phon_levenshtein) as max_avg,
    AVG(avg_phon_levenshtein) as overall_avg
FROM average_stats;
```

2. **Find optimal thresholds**:
```sql
SELECT 
    COUNT(*) FILTER (WHERE avg_phon_levenshtein <= 5.0) as very_strict,
    COUNT(*) FILTER (WHERE avg_phon_levenshtein <= 5.5) as strict,
    COUNT(*) FILTER (WHERE avg_phon_levenshtein <= 6.0) as moderate
FROM average_stats;
```

3. **Run filtered pair computation**:
```bash
.\wordpair_scorer.exe --use-filtered-words --avg-phon-threshold 5.5
```

## Estimated Timeline

- **Start**: 0h 0m - Load 75K words (< 1 second)
- **25% complete**: ~40 minutes - 18,817 words processed
- **50% complete**: ~1h 20m - 37,634 words processed
- **75% complete**: ~2h 0m - 56,450 words processed
- **100% complete**: ~2h 40m - All 75,267 words done!

Then: Analyze results and run filtered pair computation (10-50x faster than full pairs!)
