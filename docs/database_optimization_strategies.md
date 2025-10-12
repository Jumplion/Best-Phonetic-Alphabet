# Database Size Optimization Strategies

## Current State
- **Current size:** ~40.5 bytes per row
- **Full dataset:** ~104 GB for 2.83 billion pairs
- **Already applied:** 97% reduction from original 3.51 TB

## Breakdown of Current 40.5 bytes/row

| Component | Bytes | Notes |
|-----------|-------|-------|
| word_id_1 | 4-8 | SQLite INTEGER (varint encoding) |
| word_id_2 | 4-8 | SQLite INTEGER (varint encoding) |
| orth_levenshtein | 1-2 | Small INTEGER (typically 0-50) |
| phon_levenshtein | 1-2 | Small INTEGER (typically 0-50) |
| lcs_length | 1-2 | Small INTEGER (typically 0-20) |
| **SQLite overhead** | 20-25 | Headers, b-tree structure, alignment |
| **TOTAL** | **~40.5** | |

## Additional Optimization Strategies

### 1. 🔥 Bit Packing (Potential: 50-60% reduction)
**Concept:** Pack multiple small values into a single INTEGER
```sql
-- Pack all metrics into single 64-bit INTEGER:
-- word_id_1: 20 bits (supports up to 1M words)
-- word_id_2: 20 bits (supports up to 1M words)
-- orth_lev:   8 bits (0-255)
-- phon_lev:   8 bits (0-255)
-- lcs_len:    8 bits (0-255)
-- TOTAL:     64 bits = 8 bytes (vs 16-20 bytes)

CREATE TABLE word_pairs_packed (
    -- Single 64-bit packed value containing all data
    packed_data INTEGER PRIMARY KEY
) WITHOUT ROWID;
```
**Savings:** 16-20 bytes → 8 bytes data + ~15 bytes overhead = **~23 bytes/row**
**Full dataset:** 104 GB → **~65 GB** (37% reduction)

**Pros:**
- Massive space savings
- Single column = faster lookups

**Cons:**
- Requires bit manipulation in application code
- Less human-readable
- Limited to 1M words (expandable to 24-bit for 16M words)

---

### 2. 📊 Smart Filtering (Potential: 80-95% reduction)
**Concept:** Only store "interesting" pairs

**Strategy A: Distance Threshold**
```sql
-- Only store pairs with low edit distance (potential candidates)
WHERE orth_levenshtein <= 10 AND phon_levenshtein <= 10
```
- Eliminates ~80% of pairs (very dissimilar words)
- **Result:** 104 GB → **~21 GB**

**Strategy B: Same Starting Letter**
```sql
-- Only store pairs starting with same letter
WHERE word_1[0] = word_2[0]
```
- Reduces pairs by ~96% (26x fewer)
- **Result:** 104 GB → **~4 GB**

**Strategy C: Phonetic Similarity Pre-filter**
```sql
-- Only store pairs with similar phonetic structure
WHERE ABS(phoneme_count_1 - phoneme_count_2) <= 3
```
- Reduces pairs by ~70%
- **Result:** 104 GB → **~31 GB**

**Pros:**
- Dramatic size reduction
- Faster queries (fewer rows)
- May be sufficient for finding good phonetic pairs

**Cons:**
- Loses data (may miss some valid candidates)
- Need to define "interesting" criteria carefully

---

### 3. 🗜️ External Compression (Potential: 40-60% reduction)
**Concept:** Use external compressed storage

**Option A: SQLite with page-level compression**
```sql
PRAGMA page_size = 4096;
-- Use with zlib compression at OS/filesystem level
```

**Option B: Binary file with compression**
- Store as compressed binary format (protobuf, msgpack, etc.)
- Use memory-mapped files for access
- Apply gzip/zstd compression

**Savings:** 104 GB → **~40-60 GB** (depends on compression algorithm)

**Pros:**
- No schema changes
- Reversible
- Good compression ratios for repetitive data

**Cons:**
- Slower read access (decompression overhead)
- More complex application code
- Can't query directly with SQL

---

### 4. 🔄 Delta Encoding (Potential: 20-30% reduction)
**Concept:** Store differences from previous row instead of absolute values

```sql
-- Store as deltas from previous word_id
word_id_1_delta INTEGER,  -- difference from previous row
word_id_2_delta INTEGER,  -- difference from previous row
```

For sequential IDs, deltas are typically 1-10, fitting in 1 byte instead of 4-8.

**Savings:** ~5-10 bytes per row
**Full dataset:** 104 GB → **~75-85 GB**

**Pros:**
- Smaller integers = less storage
- Good for sequential access patterns

**Cons:**
- Requires sequential scanning to reconstruct
- Slower random access
- Complex to implement

---

### 5. 🎯 Hybrid Approach: Two-Tier Storage (Potential: 50-70% reduction)
**Concept:** Separate "hot" and "cold" data

**Tier 1: In-memory cache (small, fast)**
- Store only most-likely candidates (low distances)
- ~20-100M pairs = 1-4 GB

**Tier 2: Disk storage (large, compressed)**
- All other pairs in compressed format
- ~2.7B pairs compressed = 30-40 GB

**Total:** **~31-44 GB**

**Pros:**
- Best of both worlds
- Fast access to important data
- Complete coverage if needed

**Cons:**
- Complex implementation
- Need good heuristic for tier assignment

---

### 6. 💡 Statistical Sampling (Potential: 90-99% reduction)
**Concept:** Store representative sample, compute rest on-demand

**Strategy:**
- Store 1-10% of pairs uniformly sampled
- Compute other pairs on-demand when needed

**Savings:** 104 GB → **~1-10 GB**

**Pros:**
- Massive space savings
- May be statistically sufficient

**Cons:**
- Incomplete data
- Slower for full scans
- Requires recomputation

---

## Recommended Approach

### For Maximum Space Savings: **Bit Packing + Filtering**
```sql
CREATE TABLE word_pairs_optimized (
    packed_data INTEGER PRIMARY KEY,
    CHECK ((packed_data >> 48) & 0xFF <= 10)  -- orth_lev <= 10
) WITHOUT ROWID;
```
- **Expected size:** ~15-20 GB (95% reduction from current)
- **Trade-off:** Requires application-level encoding/decoding

### For Ease of Use: **Current Schema + Filtering**
```sql
-- Only store pairs with potential (low distance)
CREATE TABLE word_pairs (
    word_id_1 INTEGER NOT NULL,
    word_id_2 INTEGER NOT NULL,
    orth_levenshtein INTEGER NOT NULL CHECK (orth_levenshtein <= 10),
    phon_levenshtein INTEGER NOT NULL CHECK (phon_levenshtein <= 10),
    lcs_length INTEGER NOT NULL,
    PRIMARY KEY (word_id_1, word_id_2)
) WITHOUT ROWID;
```
- **Expected size:** ~20-30 GB (70-80% reduction from current)
- **Trade-off:** Loses high-distance pairs (usually not interesting anyway)

### For Best Balance: **Hybrid Two-Tier**
- Keep current schema for low-distance pairs
- Archive high-distance pairs separately
- **Expected size:** ~20-25 GB for active + ~30 GB archived

---

## Implementation Priority

1. **Immediate (No code changes):** Add filtering WHERE clause during insertion
2. **Short-term (Moderate effort):** Implement bit packing for 3x space savings
3. **Long-term (High effort):** Implement two-tier hybrid storage

## Conclusion

Current **104 GB** can realistically be reduced to:
- **65 GB** with bit packing (37% reduction) - moderate effort
- **21 GB** with filtering (80% reduction) - low effort
- **15 GB** with both (86% reduction) - moderate effort

The **sweet spot** is filtering + current schema → **~20-25 GB** with minimal code changes.
