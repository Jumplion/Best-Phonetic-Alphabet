#include "../include/wordpair/db_writer.h"
#include "../include/wordpair/scorer.h"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <atomic>
#include <chrono>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// Word struct implementation
// ============================================================

std::vector<std::string> Word::get_phonemes() const {
    std::vector<std::string> phonemes;
    std::istringstream iss(normalized_phoneme_list);
    std::string phoneme;
    while (iss >> phoneme) {
        phonemes.push_back(phoneme);
    }
    return phonemes;
}

// ============================================================
// DBWriter implementation
// ============================================================

struct DBWriter::Impl {
    sqlite3* db = nullptr;
};

// Helper function to execute pragma with error handling
static bool execute_pragma(sqlite3* db, const char* pragma_sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, pragma_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Warning: Failed to execute pragma '" << pragma_sql << "': " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// Helper function to begin transaction
static bool begin_transaction(sqlite3* db) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to begin transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// Helper function to commit transaction
static bool commit_transaction(sqlite3* db) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, "COMMIT;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to commit transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
    }
    return true;
}

// Helper function to pre-parse all phonemes
static std::vector<std::vector<std::string>> preparse_phonemes(const std::vector<Word>& words) {
    const size_t n = words.size();
    std::cout << "Pre-parsing phonemes for all " << n << " words...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<std::string>> all_phonemes;
    all_phonemes.reserve(n);
    for (const auto& word : words) {
        all_phonemes.push_back(word.get_phonemes());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "✅ Pre-parsed " << n << " phoneme lists in " << ms << "ms\n\n";
    
    return all_phonemes;
}

// Helper function to report progress
static void report_progress(size_t global_completed, size_t total, 
                           const std::chrono::high_resolution_clock::time_point& start_time) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    double percent = 100.0 * global_completed / total;
    double elapsed_sec = std::max(static_cast<double>(elapsed), 1.0);
    double words_per_sec = static_cast<double>(global_completed) / elapsed_sec;
    double remaining_words = total - global_completed;
    double estimated_remaining_sec = remaining_words / std::max(words_per_sec, 0.001);
    
    int eta_hours = static_cast<int>(estimated_remaining_sec / 3600);
    int eta_mins = static_cast<int>((estimated_remaining_sec - eta_hours * 3600) / 60);
    int eta_secs = static_cast<int>(estimated_remaining_sec) % 60;
    
    std::cout << "\r  Progress: " << global_completed << " / " << total 
              << " (" << std::fixed << std::setprecision(2) << percent << "%) | "
              << "Speed: " << std::setprecision(1) << words_per_sec << " words/sec | "
              << "ETA: " << eta_hours << "h " << eta_mins << "m " << eta_secs << "s    " << std::flush;
}

// ============================================================
// Helper structures for metric accumulation
// ============================================================

// Template for accumulating statistics for any numeric type
template<typename T>
struct MetricAccumulator {
    T min_val;
    T max_val;
    double total = 0.0;
    std::vector<T> values;
    
    explicit MetricAccumulator(size_t reserve_size) {
        min_val = std::numeric_limits<T>::max();
        max_val = std::numeric_limits<T>::lowest();
        values.reserve(reserve_size);
    }
    
    void add(T value) {
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
        total += static_cast<double>(value);
        values.push_back(value);
    }
};

// Helper function to assign 4-value metrics (avg, min, max, stddev) from accumulator
// MinMaxType allows for type conversion (e.g., float accumulator -> double struct fields)
template<typename AccumType, typename MinMaxType>
void assign_metric_stats(double& avg, MinMaxType& min, MinMaxType& max, double& stddev, 
                        const MetricAccumulator<AccumType>& acc, double pre_computed_avg, double pre_computed_stddev) {
    avg = pre_computed_avg;
    min = static_cast<MinMaxType>(acc.min_val);
    max = static_cast<MinMaxType>(acc.max_val);
    stddev = pre_computed_stddev;
}

DBWriter::DBWriter(const std::string& db_path) : db_path_(db_path), impl_(new Impl()) {}

DBWriter::~DBWriter() {
    if (impl_) {
        if (impl_->db) {
            sqlite3_close(impl_->db);
        }
        delete impl_;
    }
}

bool DBWriter::init() {
    int rc = sqlite3_open(db_path_.c_str(), &impl_->db);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open DB: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    // Apply performance-optimized pragmas for bulk operations
    execute_pragma(impl_->db, "PRAGMA journal_mode=WAL;");          // WAL mode for better concurrency
    execute_pragma(impl_->db, "PRAGMA synchronous=NORMAL;");        // Balance safety/speed with WAL
    execute_pragma(impl_->db, "PRAGMA cache_size=-65536;");         // 64MB cache (negative = KB)
    execute_pragma(impl_->db, "PRAGMA temp_store=MEMORY;");         // Memory for temp tables
    execute_pragma(impl_->db, "PRAGMA page_size=8192;");            // 8KB pages for large DBs
    execute_pragma(impl_->db, "PRAGMA mmap_size=268435456;");       // 256MB memory-mapped I/O

    std::cout << "✅ Database opened with performance optimizations enabled\n";
    return true;
}

std::vector<Word> DBWriter::load_all_words() const {
    std::vector<Word> words;
    
    if (!impl_ || !impl_->db) {
        std::cerr << "DBWriter not initialized. Call init() first.\n";
        return words;
    }

    const char* sql = "SELECT id, word, normalized_phoneme_list FROM words ORDER BY id;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare SQL: " << sqlite3_errmsg(impl_->db) << std::endl;
        return words;
    }

    // Reserve space for typical CMU dictionary size (~130k words)
    words.reserve(150000);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        Word w;
        w.id = sqlite3_column_int(stmt, 0);
        
        const unsigned char* word_text = sqlite3_column_text(stmt, 1);
        if (word_text) {
            w.word = reinterpret_cast<const char*>(word_text);
        }
        
        const unsigned char* phoneme_text = sqlite3_column_text(stmt, 2);
        if (phoneme_text) {
            w.normalized_phoneme_list = reinterpret_cast<const char*>(phoneme_text);
        }
        
        words.push_back(std::move(w));
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Error reading words: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    
    std::cout << "✅ Loaded " << words.size() << " words from database.\n";
    return words;
}

size_t DBWriter::batch_write_scores(const std::vector<WordPairScore>& scores) {
    if (!impl_ || !impl_->db) {
        std::cerr << "DBWriter not initialized. Call init() first.\n";
        return 0;
    }

    if (scores.empty()) {
        return 0;
    }

    // Begin transaction for batch insert
    if (!begin_transaction(impl_->db)) {
        return 0;
    }

    // Prepare INSERT statement
    // Using INSERT OR REPLACE to handle duplicates (based on PRIMARY KEY)
    // Optimized schema: uses integer IDs and lcs_length instead of text
    const char* sql = R"(
        INSERT OR REPLACE INTO word_pairs (
            word_id_1, word_id_2,
            orth_levenshtein, phon_levenshtein,
            lcs_length
        ) VALUES (?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_exec(impl_->db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return 0;
    }

    size_t inserted = 0;
    for (const auto& score : scores) {
        // Bind parameters - using integer IDs for massive space savings
        sqlite3_bind_int(stmt, 1, score.word_id_1);
        sqlite3_bind_int(stmt, 2, score.word_id_2);
        sqlite3_bind_int(stmt, 3, score.orth_levenshtein);
        sqlite3_bind_int(stmt, 4, score.phon_levenshtein);
        sqlite3_bind_int(stmt, 5, score.lcs_length);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert row: " << sqlite3_errmsg(impl_->db) << std::endl;
            // Continue with other rows rather than failing entire batch
        } else {
            inserted++;
        }

        // Reset statement for next iteration
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    // Commit transaction
    if (!commit_transaction(impl_->db)) {
        return 0;
    }

    return inserted;
}

size_t DBWriter::batch_write_average_stats(const std::vector<WordAverageStats>& stats) {
    if (!impl_ || !impl_->db) {
        std::cerr << "DBWriter not initialized. Call init() first.\n";
        return 0;
    }

    if (stats.empty()) {
        return 0;
    }

    // Begin transaction for batch insert
    if (!begin_transaction(impl_->db)) {
        return 0;
    }

    // Prepare INSERT statement
    const char* sql = R"(
        INSERT OR REPLACE INTO average_stats (
            word_id,
            avg_orth_levenshtein, avg_phon_levenshtein, avg_lcs_length,
            min_orth_levenshtein, max_orth_levenshtein, stddev_orth_levenshtein,
            min_phon_levenshtein, max_phon_levenshtein, stddev_phon_levenshtein,
            avg_weighted_phon_levenshtein, min_weighted_phon_levenshtein, 
            max_weighted_phon_levenshtein, stddev_weighted_phon_levenshtein,
            count_close_weighted_phon,
            avg_audio_phon_levenshtein, min_audio_phon_levenshtein,
            max_audio_phon_levenshtein, stddev_audio_phon_levenshtein,
            count_close_audio_phon,
            avg_orth_jaccard, min_orth_jaccard, max_orth_jaccard, stddev_orth_jaccard,
            avg_phon_jaccard, min_phon_jaccard, max_phon_jaccard, stddev_phon_jaccard,
            count_close_orth, count_close_phon
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_exec(impl_->db, "ROLLBACK;", nullptr, nullptr, nullptr);
        return 0;
    }

    size_t inserted = 0;
    for (const auto& stat : stats) {
        // Bind all parameters
        sqlite3_bind_int(stmt, 1, stat.word_id);
        sqlite3_bind_double(stmt, 2, stat.avg_orth_levenshtein);
        sqlite3_bind_double(stmt, 3, stat.avg_phon_levenshtein);
        sqlite3_bind_double(stmt, 4, stat.avg_lcs_length);
        sqlite3_bind_int(stmt, 5, stat.min_orth_levenshtein);
        sqlite3_bind_int(stmt, 6, stat.max_orth_levenshtein);
        sqlite3_bind_double(stmt, 7, stat.stddev_orth_levenshtein);
        sqlite3_bind_int(stmt, 8, stat.min_phon_levenshtein);
        sqlite3_bind_int(stmt, 9, stat.max_phon_levenshtein);
        sqlite3_bind_double(stmt, 10, stat.stddev_phon_levenshtein);
        sqlite3_bind_double(stmt, 11, stat.avg_weighted_phon_levenshtein);
        sqlite3_bind_double(stmt, 12, stat.min_weighted_phon_levenshtein);
        sqlite3_bind_double(stmt, 13, stat.max_weighted_phon_levenshtein);
        sqlite3_bind_double(stmt, 14, stat.stddev_weighted_phon_levenshtein);
        sqlite3_bind_int(stmt, 15, stat.count_close_weighted_phon);
        sqlite3_bind_double(stmt, 16, stat.avg_audio_phon_levenshtein);
        sqlite3_bind_double(stmt, 17, stat.min_audio_phon_levenshtein);
        sqlite3_bind_double(stmt, 18, stat.max_audio_phon_levenshtein);
        sqlite3_bind_double(stmt, 19, stat.stddev_audio_phon_levenshtein);
        sqlite3_bind_int(stmt, 20, stat.count_close_audio_phon);
        sqlite3_bind_double(stmt, 21, stat.avg_orth_jaccard);
        sqlite3_bind_double(stmt, 22, stat.min_orth_jaccard);
        sqlite3_bind_double(stmt, 23, stat.max_orth_jaccard);
        sqlite3_bind_double(stmt, 24, stat.stddev_orth_jaccard);
        sqlite3_bind_double(stmt, 25, stat.avg_phon_jaccard);
        sqlite3_bind_double(stmt, 26, stat.min_phon_jaccard);
        sqlite3_bind_double(stmt, 27, stat.max_phon_jaccard);
        sqlite3_bind_double(stmt, 28, stat.stddev_phon_jaccard);
        sqlite3_bind_int(stmt, 29, stat.count_close_orth);
        sqlite3_bind_int(stmt, 30, stat.count_close_phon);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert stats row: " << sqlite3_errmsg(impl_->db) << std::endl;
        } else {
            inserted++;
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    // Commit transaction
    if (!commit_transaction(impl_->db)) {
        return 0;
    }

    return inserted;
}

std::vector<Word> DBWriter::load_filtered_words(
    double max_avg_orth_lev,
    double max_avg_phon_lev,
    int min_close_matches
) const {
    std::vector<Word> words;
    
    if (!impl_ || !impl_->db) {
        std::cerr << "DBWriter not initialized. Call init() first.\n";
        return words;
    }

    // Join words and average_stats tables to get filtered words
    const char* sql = R"(
        SELECT w.id, w.word, w.normalized_phoneme_list
        FROM words w
        INNER JOIN average_stats a ON w.id = a.word_id
        WHERE a.avg_orth_levenshtein <= ?
          AND a.avg_phon_levenshtein <= ?
          AND a.count_close_phon >= ?
        ORDER BY w.id;
    )";
    
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare SQL: " << sqlite3_errmsg(impl_->db) << std::endl;
        return words;
    }

    // Bind filter parameters
    sqlite3_bind_double(stmt, 1, max_avg_orth_lev);
    sqlite3_bind_double(stmt, 2, max_avg_phon_lev);
    sqlite3_bind_int(stmt, 3, min_close_matches);

    words.reserve(10000); // Reserve space for filtered results

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        Word w;
        w.id = sqlite3_column_int(stmt, 0);
        
        const unsigned char* word_text = sqlite3_column_text(stmt, 1);
        if (word_text) {
            w.word = reinterpret_cast<const char*>(word_text);
        }
        
        const unsigned char* phoneme_text = sqlite3_column_text(stmt, 2);
        if (phoneme_text) {
            w.normalized_phoneme_list = reinterpret_cast<const char*>(phoneme_text);
        }
        
        words.push_back(std::move(w));
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Error reading filtered words: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    
    std::cout << "✅ Loaded " << words.size() << " filtered words from database.\n";
    return words;
}

size_t DBWriter::compute_and_write_average_stats_batched(
    const std::vector<Word>& words,
    size_t batch_size,
    int orth_close_threshold,
    int phon_close_threshold
) {
    const size_t n = words.size();
    if (n == 0) {
        std::cout << "No words to process.\n";
        return 0;
    }

    std::cout << "Computing and writing average statistics in batches...\n";
    std::cout << "Total words: " << n << " | Batch size: " << batch_size << "\n";
    std::cout << "Progress updates every word...\n\n";

    auto overall_start = std::chrono::high_resolution_clock::now();
    size_t total_written = 0;
    
    // PRE-PARSE OPTIMIZATION: Parse all phonemes once upfront
    // This avoids repeated string splitting in the inner loop (1.5-2× speedup)
    std::vector<std::vector<std::string>> all_phonemes = preparse_phonemes(words);
    
    // Process in batches
    for (size_t batch_start = 0; batch_start < n; batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, n);
        size_t current_batch_size = batch_end - batch_start;
        
        std::cout << "\n--- Batch: words " << batch_start << " to " << (batch_end - 1) 
                  << " (" << current_batch_size << " words) ---\n";
        
        // Compute stats for this batch
        std::vector<WordAverageStats> batch_stats(current_batch_size);
        
        auto batch_compute_start = std::chrono::high_resolution_clock::now();
        std::atomic<size_t> completed_in_batch(0);

        #pragma omp parallel for schedule(dynamic, 10)
        // Outerloop: parallelized over words in the current batch
        for (size_t i = batch_start; i < batch_end; ++i) {
            const Word& word_i = words[i];
            const auto& phonemes_i = all_phonemes[i];  // Use pre-parsed phonemes

            // Initialize metric accumulators (reserve n-1 since we skip i==j)
            const size_t n_comparisons = n - 1;
            MetricAccumulator<int> orth_acc(n_comparisons);
            MetricAccumulator<int> phon_acc(n_comparisons);
            MetricAccumulator<float> weighted_phon_acc(n_comparisons);
            MetricAccumulator<float> audio_phon_acc(n_comparisons);
            MetricAccumulator<float> orth_jaccard_acc(n_comparisons);
            MetricAccumulator<float> phon_jaccard_acc(n_comparisons);
            
            long long total_lcs = 0;
            int count_close_orth = 0;
            int count_close_phon = 0;
            int count_close_weighted_phon = 0;
            int count_close_audio_phon = 0;

            // First pass: compute means and collect distances
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                const Word& word_j = words[j];
                const auto& phonemes_j = all_phonemes[j];  // Use pre-parsed phonemes

                // Compute distances
                int orth_dist = orthographic_levenshtein_score(word_i.word, word_j.word);

                // Compute both weighted and audio distances in a single pass (2x faster!)
                int phon_dist;
                float weighted_phon_dist, audio_phon_dist;
                phonetic_levenshtein_scores(phonemes_i, phonemes_j, phon_dist, weighted_phon_dist, audio_phon_dist);
                
                // Compute Jaccard indices
                float orth_jaccard = orthographic_jaccard_index(word_i.word, word_j.word);
                float phon_jaccard = phonetic_jaccard_index(phonemes_i, phonemes_j);
                
                // TEMPORARILY DISABLED - LCS computation commented out for performance testing
                // auto lcs_seqs = longest_contiguous_subsequence(phonemes_i, phonemes_j);
                // int lcs = lcs_seqs.empty() ? 0 : lcs_seqs[0].size();
                int lcs = -1;  // Placeholder value while LCS is disabled

                // Accumulate metrics (stores value and tracks min/max/total automatically)
                orth_acc.add(orth_dist);
                phon_acc.add(phon_dist);
                weighted_phon_acc.add(weighted_phon_dist);
                audio_phon_acc.add(audio_phon_dist);
                orth_jaccard_acc.add(orth_jaccard);
                phon_jaccard_acc.add(phon_jaccard);
                
                total_lcs += lcs;  // Will accumulate -1 values (placeholder)

                // Count close matches
                if (orth_dist <= orth_close_threshold) {
                    count_close_orth++;
                }
                if (phon_dist <= phon_close_threshold) {
                    count_close_phon++;
                }
                
                // Use a reasonable threshold for weighted/audio distances (e.g., 30% of max phoneme count)
                float dynamic_threshold = static_cast<float>(std::max(phonemes_i.size(), phonemes_j.size())) * 0.3f;
                if (weighted_phon_dist <= dynamic_threshold) {
                    count_close_weighted_phon++;
                }
                if (audio_phon_dist <= dynamic_threshold) {
                    count_close_audio_phon++;
                }
            }
 
            // Compute averages
            double avg_orth = orth_acc.total / n_comparisons;
            double avg_phon = phon_acc.total / n_comparisons;
            double avg_weighted_phon = weighted_phon_acc.total / n_comparisons;
            double avg_audio_phon = audio_phon_acc.total / n_comparisons;
            double avg_orth_jaccard = orth_jaccard_acc.total / n_comparisons;
            double avg_phon_jaccard = phon_jaccard_acc.total / n_comparisons;
            
            // Compute all standard deviations in a single pass for better cache efficiency
            double sum_sq_diff_orth = 0.0;
            double sum_sq_diff_phon = 0.0;
            double sum_sq_diff_weighted_phon = 0.0;
            double sum_sq_diff_audio_phon = 0.0;
            double sum_sq_diff_orth_jaccard = 0.0;
            double sum_sq_diff_phon_jaccard = 0.0;
            
            for (size_t k = 0; k < n_comparisons; ++k) {
                double diff_orth = orth_acc.values[k] - avg_orth;
                double diff_phon = phon_acc.values[k] - avg_phon;
                double diff_weighted_phon = weighted_phon_acc.values[k] - avg_weighted_phon;
                double diff_audio_phon = audio_phon_acc.values[k] - avg_audio_phon;
                double diff_orth_jaccard = orth_jaccard_acc.values[k] - avg_orth_jaccard;
                double diff_phon_jaccard = phon_jaccard_acc.values[k] - avg_phon_jaccard;
                
                sum_sq_diff_orth += diff_orth * diff_orth;
                sum_sq_diff_phon += diff_phon * diff_phon;
                sum_sq_diff_weighted_phon += diff_weighted_phon * diff_weighted_phon;
                sum_sq_diff_audio_phon += diff_audio_phon * diff_audio_phon;
                sum_sq_diff_orth_jaccard += diff_orth_jaccard * diff_orth_jaccard;
                sum_sq_diff_phon_jaccard += diff_phon_jaccard * diff_phon_jaccard;
            }
            
            double stddev_orth = std::sqrt(sum_sq_diff_orth / n_comparisons);
            double stddev_phon = std::sqrt(sum_sq_diff_phon / n_comparisons);
            double stddev_weighted_phon = std::sqrt(sum_sq_diff_weighted_phon / n_comparisons);
            double stddev_audio_phon = std::sqrt(sum_sq_diff_audio_phon / n_comparisons);
            double stddev_orth_jaccard = std::sqrt(sum_sq_diff_orth_jaccard / n_comparisons);
            double stddev_phon_jaccard = std::sqrt(sum_sq_diff_phon_jaccard / n_comparisons);

            // Store results using helper function to reduce repetition
            size_t batch_index = i - batch_start;
            WordAverageStats& stat = batch_stats[batch_index];
            stat.word_id = word_i.id;
            
            // Assign all 4-value metrics (avg, min, max, stddev) using helper
            assign_metric_stats(stat.avg_orth_levenshtein, stat.min_orth_levenshtein, 
                              stat.max_orth_levenshtein, stat.stddev_orth_levenshtein, 
                              orth_acc, avg_orth, stddev_orth);
            
            assign_metric_stats(stat.avg_phon_levenshtein, stat.min_phon_levenshtein, 
                              stat.max_phon_levenshtein, stat.stddev_phon_levenshtein, 
                              phon_acc, avg_phon, stddev_phon);
            
            assign_metric_stats(stat.avg_weighted_phon_levenshtein, stat.min_weighted_phon_levenshtein, 
                              stat.max_weighted_phon_levenshtein, stat.stddev_weighted_phon_levenshtein, 
                              weighted_phon_acc, avg_weighted_phon, stddev_weighted_phon);
            
            assign_metric_stats(stat.avg_audio_phon_levenshtein, stat.min_audio_phon_levenshtein, 
                              stat.max_audio_phon_levenshtein, stat.stddev_audio_phon_levenshtein, 
                              audio_phon_acc, avg_audio_phon, stddev_audio_phon);
            
            assign_metric_stats(stat.avg_orth_jaccard, stat.min_orth_jaccard, 
                              stat.max_orth_jaccard, stat.stddev_orth_jaccard, 
                              orth_jaccard_acc, avg_orth_jaccard, stddev_orth_jaccard);
            
            assign_metric_stats(stat.avg_phon_jaccard, stat.min_phon_jaccard, 
                              stat.max_phon_jaccard, stat.stddev_phon_jaccard, 
                              phon_jaccard_acc, avg_phon_jaccard, stddev_phon_jaccard);
            
            // Assign close match counts
            stat.count_close_orth = count_close_orth;
            stat.count_close_phon = count_close_phon;
            stat.count_close_weighted_phon = count_close_weighted_phon;
            stat.count_close_audio_phon = count_close_audio_phon;
            
            // LCS (placeholder)
            stat.avg_lcs_length = static_cast<double>(total_lcs) / n_comparisons;

            // Progress reporting
            size_t current_completed = ++completed_in_batch;
            size_t global_completed = batch_start + current_completed;
            
            #pragma omp critical
            {
                report_progress(global_completed, n, overall_start);
            }
        }

        auto batch_compute_end = std::chrono::high_resolution_clock::now();
        auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_compute_end - batch_compute_start).count();
        
        std::cout << "\n  Batch computation took " << (compute_ms / 1000.0) << " seconds\n";

        // Write batch to database
        std::cout << "  Writing batch to database...\n";
        auto write_start = std::chrono::high_resolution_clock::now();
        size_t written = batch_write_average_stats(batch_stats);
        auto write_end = std::chrono::high_resolution_clock::now();
        auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start).count();
        
        std::cout << "  ✅ Wrote " << written << " rows in " << write_ms << " ms\n";
        total_written += written;
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(overall_end - overall_start).count();
    int hours = total_seconds / 3600;
    int mins = (total_seconds % 3600) / 60;
    int secs = total_seconds % 60;
    
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "✅ Completed all batches!\n";
    std::cout << "   Total words processed: " << n << "\n";
    std::cout << "   Total rows written: " << total_written << "\n";
    std::cout << "   Total time: " << hours << "h " << mins << "m " << secs << "s\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    return total_written;
}
