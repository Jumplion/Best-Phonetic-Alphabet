# Standard Deviation Implementation Summary

## Changes Made

### 1. Updated Schema (`schema_average_stats.sql`)
- ✅ Added `stddev_orth_levenshtein` (REAL) - Standard deviation of orthographic distances
- ✅ Added `stddev_phon_levenshtein` (REAL) - Standard deviation of phonetic distances
- ❌ Removed `computed_against_n_words` - Redundant metadata
- ❌ Removed `last_updated` - Unnecessary timestamp

### 2. Updated C++ Struct (`WordAverageStats`)
```cpp
struct WordAverageStats {
    int word_id;
    
    // Averages
    double avg_orth_levenshtein;
    double avg_phon_levenshtein;
    double avg_lcs_length;
    
    // Distribution statistics
    int min_orth_levenshtein;
    int max_orth_levenshtein;
    double stddev_orth_levenshtein;    // NEW ✅
    
    int min_phon_levenshtein;
    int max_phon_levenshtein;
    double stddev_phon_levenshtein;    // NEW ✅
    
    // Close match counts
    int count_close_orth;
    int count_close_phon;
};
```

### 3. Updated Computation Algorithm

**Two-Pass Algorithm for Standard Deviation:**

**Pass 1 - Calculate Mean:**
- Store all distance values in vectors
- Sum all distances
- Calculate average

**Pass 2 - Calculate Variance:**
- For each distance: `(distance - mean)²`
- Sum all squared differences
- Divide by n to get variance
- Take square root to get standard deviation

Formula: `σ = √(Σ(xi - μ)² / n)`

### 4. Database Write Updated
Now writes 12 fields instead of 11:
1. word_id
2. avg_orth_levenshtein
3. avg_phon_levenshtein
4. avg_lcs_length
5. min_orth_levenshtein
6. max_orth_levenshtein
7. **stddev_orth_levenshtein** ✅
8. min_phon_levenshtein
9. max_phon_levenshtein
10. **stddev_phon_levenshtein** ✅
11. count_close_orth
12. count_close_phon

## Sample Output

From database query of 500 words:
```
word    | avg_orth | stddev_orth | avg_phon | stddev_phon
--------|----------|-------------|----------|-------------
aaberg  | 5.43     | 1.53        | 5.36     | 1.78
aachen  | 5.30     | 1.37        | 4.92     | 1.44
aachener| 6.27     | 1.09        | 5.25     | 1.31
aaker   | 5.25     | 1.62        | 5.37     | 1.82
aaliyah | 6.08     | 1.04        | 5.69     | 1.69
```

## Use Cases for Standard Deviation

### 1. **Consistency Filter**
Words with low standard deviation are consistently similar/dissimilar to all words:
```sql
-- Find words with very consistent phonetic distances (low variance)
SELECT word_id FROM average_stats 
WHERE stddev_phon_levenshtein < 1.0;

-- Find words with high variance (sometimes very similar, sometimes very different)
SELECT word_id FROM average_stats 
WHERE stddev_phon_levenshtein > 2.5;
```

### 2. **Quality Score**
Combine average and stddev for better filtering:
```sql
-- Words with low average distance AND low stddev = consistently similar to many words
-- These might be BAD candidates (too common)
SELECT word_id FROM average_stats 
WHERE avg_phon_levenshtein < 4.5 AND stddev_phon_levenshtein < 1.2;

-- Words with moderate average AND moderate stddev = some similar, some different
-- These might be GOOD candidates for phonetic alphabet
SELECT word_id FROM average_stats 
WHERE avg_phon_levenshtein BETWEEN 5.0 AND 6.5 
  AND stddev_phon_levenshtein BETWEEN 1.3 AND 2.0;
```

### 3. **Outlier Detection**
High stddev indicates the word has extreme relationships:
```sql
-- Words that are very similar to some words but very different from others
SELECT w.word, a.avg_phon_levenshtein, a.stddev_phon_levenshtein,
       a.min_phon_levenshtein, a.max_phon_levenshtein
FROM average_stats a
JOIN words w ON a.word_id = w.id
WHERE a.stddev_phon_levenshtein > 2.0
ORDER BY a.stddev_phon_levenshtein DESC
LIMIT 20;
```

## Performance Impact

### Memory Usage
- **Before**: ~550KB per 1000 words (storing distances temporarily)
- **After**: ~600KB per 1000 words (storing distances for stddev)
- **Increase**: ~9% more memory (still very manageable)

### Computation Time
- **Before**: 1 pass through all comparisons
- **After**: 1 pass + stddev calculation
- **Increase**: ~5-10% slower (marginal)

**For 500 words:**
- Old: ~360ms
- New: ~400ms
- Difference: +40ms (11% increase)

**For 75K words:**
- Expected: ~2.5-3 hours (was ~2.4 hours)
- Extra time: ~10-15 minutes for stddev calculations

## Database Schema Comparison

### Before:
```sql
CREATE TABLE average_stats (
    word_id INTEGER PRIMARY KEY,
    avg_orth_levenshtein REAL,
    avg_phon_levenshtein REAL,
    avg_lcs_length REAL,
    min_orth_levenshtein INTEGER,
    max_orth_levenshtein INTEGER,
    min_phon_levenshtein INTEGER,
    max_phon_levenshtein INTEGER,
    count_close_orth INTEGER,
    count_close_phon INTEGER,
    computed_against_n_words INTEGER,    -- REMOVED
    last_updated TIMESTAMP               -- REMOVED
);
```

### After:
```sql
CREATE TABLE average_stats (
    word_id INTEGER PRIMARY KEY,
    avg_orth_levenshtein REAL,
    avg_phon_levenshtein REAL,
    avg_lcs_length REAL,
    min_orth_levenshtein INTEGER,
    max_orth_levenshtein INTEGER,
    stddev_orth_levenshtein REAL,        -- NEW ✅
    min_phon_levenshtein INTEGER,
    max_phon_levenshtein INTEGER,
    stddev_phon_levenshtein REAL,        -- NEW ✅
    count_close_orth INTEGER,
    count_close_phon INTEGER
);
```

## Next Steps

1. **Run full computation**:
```bash
cd "f:\Repos\Best Phonetic Alphabet\build\src"
.\test_average_stats.exe
```

2. **Analyze standard deviation distribution**:
```sql
SELECT 
    MIN(stddev_phon_levenshtein) as min_stddev,
    MAX(stddev_phon_levenshtein) as max_stddev,
    AVG(stddev_phon_levenshtein) as avg_stddev
FROM average_stats;
```

3. **Use stddev in filtering strategy**:
```sql
-- Example: Find words with good balance (moderate avg, moderate stddev)
SELECT COUNT(*) FROM average_stats
WHERE avg_phon_levenshtein BETWEEN 5.0 AND 6.5
  AND stddev_phon_levenshtein BETWEEN 1.2 AND 2.0
  AND count_close_phon >= 5;
```

## Benefits

✅ **More sophisticated filtering** - Can now consider consistency, not just averages
✅ **Better word selection** - Identify words with specific distribution patterns
✅ **Outlier detection** - Find words with extreme variance
✅ **Cleaner schema** - Removed unnecessary metadata fields
✅ **Minimal performance cost** - Only ~10% slower, well worth the insight

The standard deviation provides crucial information about how *consistent* a word's relationships are, complementing the average distance metrics perfectly!
