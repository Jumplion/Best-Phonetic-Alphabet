#include "../include/wordpair/db_writer.h"
#include "../include/wordpair/scorer.h"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

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
    char* err_msg = nullptr;
    int rc = sqlite3_exec(impl_->db, "BEGIN TRANSACTION;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to begin transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
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
    rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
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
    rc = sqlite3_exec(impl_->db, "COMMIT;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to commit transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_exec(impl_->db, "ROLLBACK;", nullptr, nullptr, nullptr);
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
    char* err_msg = nullptr;
    int rc = sqlite3_exec(impl_->db, "BEGIN TRANSACTION;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to begin transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return 0;
    }

    // Prepare INSERT statement
    const char* sql = R"(
        INSERT OR REPLACE INTO average_stats (
            word_id,
            avg_orth_levenshtein, avg_phon_levenshtein, avg_lcs_length,
            min_orth_levenshtein, max_orth_levenshtein,
            min_phon_levenshtein, max_phon_levenshtein,
            count_close_orth, count_close_phon,
            computed_against_n_words
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
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
        sqlite3_bind_int(stmt, 7, stat.min_phon_levenshtein);
        sqlite3_bind_int(stmt, 8, stat.max_phon_levenshtein);
        sqlite3_bind_int(stmt, 9, stat.count_close_orth);
        sqlite3_bind_int(stmt, 10, stat.count_close_phon);
        sqlite3_bind_int(stmt, 11, stat.computed_against_n_words);

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
    rc = sqlite3_exec(impl_->db, "COMMIT;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to commit transaction: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_exec(impl_->db, "ROLLBACK;", nullptr, nullptr, nullptr);
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

std::vector<WordAverageStats> DBWriter::compute_average_stats(
    const std::vector<Word>& words,
    int orth_close_threshold,
    int phon_close_threshold
) {
    const size_t n = words.size();
    std::vector<WordAverageStats> stats(n);

    std::cout << "Computing average statistics for " << n << " words...\n";

    // Parallel computation of statistics
    #pragma omp parallel for schedule(dynamic, 10)
    for (size_t i = 0; i < n; ++i) {
        const Word& word_i = words[i];
        auto phonemes_i = word_i.get_phonemes();

        // Accumulators for this word
        long long total_orth = 0;
        long long total_phon = 0;
        long long total_lcs = 0;
        int min_orth = INT_MAX;
        int max_orth = INT_MIN;
        int min_phon = INT_MAX;
        int max_phon = INT_MIN;
        int count_close_orth = 0;
        int count_close_phon = 0;

        // Compare against all other words
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue; // Skip self-comparison

            const Word& word_j = words[j];
            auto phonemes_j = word_j.get_phonemes();

            // Compute distances
            int orth_dist = orthographic_levenshtein_score(word_i.word, word_j.word);
            int phon_dist = phonetic_levenshtein_score(phonemes_i, phonemes_j);
            auto lcs_seqs = longest_contiguous_subsequence(phonemes_i, phonemes_j);
            int lcs = lcs_seqs.empty() ? 0 : lcs_seqs[0].size();

            // Accumulate totals
            total_orth += orth_dist;
            total_phon += phon_dist;
            total_lcs += lcs;

            // Track min/max
            min_orth = std::min(min_orth, orth_dist);
            max_orth = std::max(max_orth, orth_dist);
            min_phon = std::min(min_phon, phon_dist);
            max_phon = std::max(max_phon, phon_dist);

            // Count close matches
            if (orth_dist <= orth_close_threshold) {
                count_close_orth++;
            }
            if (phon_dist <= phon_close_threshold) {
                count_close_phon++;
            }
        }

        // Compute averages (n-1 comparisons per word)
        const int n_comparisons = n - 1;
        WordAverageStats& stat = stats[i];
        stat.word_id = word_i.id;
        stat.avg_orth_levenshtein = static_cast<double>(total_orth) / n_comparisons;
        stat.avg_phon_levenshtein = static_cast<double>(total_phon) / n_comparisons;
        stat.avg_lcs_length = static_cast<double>(total_lcs) / n_comparisons;
        stat.min_orth_levenshtein = min_orth;
        stat.max_orth_levenshtein = max_orth;
        stat.min_phon_levenshtein = min_phon;
        stat.max_phon_levenshtein = max_phon;
        stat.count_close_orth = count_close_orth;
        stat.count_close_phon = count_close_phon;
        stat.computed_against_n_words = n_comparisons;

        // Progress reporting (every 1000 words)
        if (i > 0 && i % 1000 == 0) {
            #pragma omp critical
            {
                std::cout << "  Processed " << i << " / " << n << " words ("
                          << (100.0 * i / n) << "%)\n";
            }
        }
    }

    std::cout << "✅ Computed average statistics for " << n << " words.\n";
    return stats;
}