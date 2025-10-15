#include "../include/wordpair/scorer.h"
#include <algorithm>
#include <cctype>
#include <sqlite3.h>
#include <unordered_map>
#include <mutex>
#include <iostream>

// File-scope phoneme distance caches
// These maps are populated once during initialization and then treated as read-only,
// making them safe for concurrent access from multiple threads without locking.
static std::unordered_map<std::string, float> phoneme_distance_map;  // feature-based distances
static std::unordered_map<std::string, float> phoneme_audio_distance_map;  // audio-based distances
// Mutex only used during initialization to ensure safe map updates
static std::mutex phoneme_map_mutex;

// Helper: validate that phonemes don't contain stress digits
static void validate_normalized_phonemes(const std::vector<std::string>& phonemes, const char* sequence_name) {
    for (const auto& phoneme : phonemes) {
        for (char c : phoneme) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                throw std::invalid_argument(
                    "Non-normalized phoneme '" + phoneme + "' found in sequence '" + sequence_name + "'. "
                    "Please pass normalized phonemes without stress digits (e.g., 'AH' instead of 'AH1')."
                );
            }
        }
    }
}

// Helper: generic preload function for phoneme distances
static bool preload_phoneme_distances_impl(
    const std::string& db_path,
    const char* column_name,
    std::unordered_map<std::string, float>& target_map
) {
    sqlite3* db = nullptr;
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        if (db) {
            std::cerr << "Failed to open phoneme DB: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_close(db);
        } else {
            std::cerr << "Failed to open phoneme DB (null handle)\n";
        }
        return false;
    }

    std::string sql = "SELECT phoneme_1, phoneme_2, " + std::string(column_name) + " FROM phoneme_distances;";
    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare phoneme distances query (" << column_name << "): " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }

    std::unordered_map<std::string, float> local_map;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char* p1 = sqlite3_column_text(stmt, 0);
        const unsigned char* p2 = sqlite3_column_text(stmt, 1);
        double dist = sqlite3_column_double(stmt, 2);
        if (!p1 || !p2) continue;
        std::string s1 = reinterpret_cast<const char*>(p1);
        std::string s2 = reinterpret_cast<const char*>(p2);
        std::string key = s1 + "|" + s2;
        local_map[key] = static_cast<float>(dist);
        local_map[s2 + "|" + s1] = static_cast<float>(dist);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    {
        std::lock_guard<std::mutex> lk(phoneme_map_mutex);
        target_map = std::move(local_map);
    }
    return true;
}

int orthographic_levenshtein_score(const std::string& a, const std::string& b) {
    std::string a_lower = a;
    std::string b_lower = b;
    std::transform(a_lower.begin(), a_lower.end(), a_lower.begin(), [](unsigned char c){ return std::tolower(c); });
    std::transform(b_lower.begin(), b_lower.end(), b_lower.begin(), [](unsigned char c){ return std::tolower(c); });

    size_t len_a = a_lower.size();
    size_t len_b = b_lower.size();
    
    // Handle edge cases
    if (len_a == 0) return static_cast<int>(len_b);
    if (len_b == 0) return static_cast<int>(len_a);
    
    // Space-optimized: use only two rows instead of full 2D matrix
    // This reduces memory from O(n*m) to O(m), improving cache locality
    std::vector<int> prev_row(len_b + 1);
    std::vector<int> curr_row(len_b + 1);

    // Initialize first row
    for (size_t j = 0; j <= len_b; ++j) {
        prev_row[j] = static_cast<int>(j);
    }

    // Process each character in string a
    for (size_t i = 1; i <= len_a; ++i) {
        curr_row[0] = static_cast<int>(i);
        
        for (size_t j = 1; j <= len_b; ++j) {
            int cost = (a_lower[i - 1] == b_lower[j - 1]) ? 0 : 1;
            curr_row[j] = std::min({ prev_row[j] + 1,         // Deletion
                                     curr_row[j - 1] + 1,     // Insertion
                                     prev_row[j - 1] + cost });  // Substitution
        }
        
        // Swap rows for next iteration
        std::swap(prev_row, curr_row);
    }
    
    return prev_row[len_b];
}

int phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t len_a = a.size();
    size_t len_b = b.size();
    
    // Handle edge cases
    if (len_a == 0) return static_cast<int>(len_b);
    if (len_b == 0) return static_cast<int>(len_a);
    
    // Space-optimized: use only two rows instead of full 2D matrix
    std::vector<int> prev_row(len_b + 1);
    std::vector<int> curr_row(len_b + 1);

    // Initialize first row
    for (size_t j = 0; j <= len_b; ++j) {
        prev_row[j] = static_cast<int>(j);
    }

    // Process each phoneme in sequence a
    for (size_t i = 1; i <= len_a; ++i) {
        curr_row[0] = static_cast<int>(i);
        
        for (size_t j = 1; j <= len_b; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            curr_row[j] = std::min({ prev_row[j] + 1,         // Deletion
                                     curr_row[j - 1] + 1,     // Insertion
                                     prev_row[j - 1] + cost });  // Substitution
        }
        
        // Swap rows for next iteration
        std::swap(prev_row, curr_row);
    }
    
    return prev_row[len_b];
}

void combined_phonetic_levenshtein_scores(
    const std::vector<std::string>& a, 
    const std::vector<std::string>& b,
    float& out_weighted_score,
    float& out_audio_score
) {
    // Phoneme distance lookups for both maps
    // Use thread-local static buffers to eliminate millions of string allocations
    auto phoneme_distance_weighted = [&](const std::string &p1, const std::string &p2) -> float {
        if (p1 == p2) return 0.0f;
        
        // Thread-local buffer: allocated once per thread, reused for all calls
        // This eliminates ~100k+ allocations in typical workloads
        thread_local std::string key;
        key.clear();
        key.reserve(p1.size() + p2.size() + 1);
        key = p1; key += '|'; key += p2;
        auto it = phoneme_distance_map.find(key);
        if (it != phoneme_distance_map.end()) return it->second;
        
        // Reuse same buffer for reverse lookup
        key.clear();
        key = p2; key += '|'; key += p1;
        it = phoneme_distance_map.find(key);
        if (it != phoneme_distance_map.end()) return it->second;

        throw std::runtime_error(
            "Missing phoneme distance entry in feature-based distance map for phoneme pair ('" + 
            p1 + "', '" + p2 + "'). Please ensure preload_phoneme_feature_distances was called."
        );
    };

    auto phoneme_distance_audio = [&](const std::string &p1, const std::string &p2) -> float {
        if (p1 == p2) return 0.0f;
        
        // Thread-local buffer: allocated once per thread, reused for all calls
        thread_local std::string key;
        key.clear();
        key.reserve(p1.size() + p2.size() + 1);
        key = p1; key += '|'; key += p2;
        auto it = phoneme_audio_distance_map.find(key);
        if (it != phoneme_audio_distance_map.end()) return it->second;
        
        // Reuse same buffer for reverse lookup
        key.clear();
        key = p2; key += '|'; key += p1;
        it = phoneme_audio_distance_map.find(key);
        if (it != phoneme_audio_distance_map.end()) return it->second;

        throw std::runtime_error(
            "Missing phoneme distance entry in audio-based distance map for phoneme pair ('" + 
            p1 + "', '" + p2 + "'). Please ensure preload_phoneme_audio_distances was called."
        );
    };

    const float GAP_BASE = 0.6f;
    const float GAP_PHONEME_SCALE = 0.8f;

    size_t len_a = a.size();
    size_t len_b = b.size();
    
    if (len_a == 0) {
        out_weighted_score = len_b * GAP_BASE;
        out_audio_score = len_b * GAP_BASE;
        return;
    }
    if (len_b == 0) {
        out_weighted_score = len_a * GAP_BASE;
        out_audio_score = len_a * GAP_BASE;
        return;
    }

    // Separate DP rows for each distance metric
    std::vector<float> prev_row_weighted(len_b + 1);
    std::vector<float> curr_row_weighted(len_b + 1);
    std::vector<float> prev_row_audio(len_b + 1);
    std::vector<float> curr_row_audio(len_b + 1);

    // Initialize first rows
    prev_row_weighted[0] = 0.0f;
    prev_row_audio[0] = 0.0f;
    for (size_t j = 1; j <= len_b; ++j) {
        prev_row_weighted[j] = prev_row_weighted[j-1] + GAP_BASE;
        prev_row_audio[j] = prev_row_audio[j-1] + GAP_BASE;
    }

    // Process both algorithms simultaneously
    for (size_t i = 1; i <= len_a; ++i) {
        curr_row_weighted[0] = prev_row_weighted[0] + GAP_BASE;
        curr_row_audio[0] = prev_row_audio[0] + GAP_BASE;
        
        for (size_t j = 1; j <= len_b; ++j) {
            // Compute distances once for this phoneme pair
            float phon_dist_weighted = phoneme_distance_weighted(a[i-1], b[j-1]);
            float phon_dist_audio = phoneme_distance_audio(a[i-1], b[j-1]);

            // Weighted (feature-based) calculation
            float sub_cost_w = phon_dist_weighted;
            float del_cost_w = GAP_BASE + GAP_PHONEME_SCALE * phon_dist_weighted;
            float ins_cost_w = GAP_BASE + GAP_PHONEME_SCALE * phon_dist_weighted;
            curr_row_weighted[j] = std::min({ prev_row_weighted[j] + del_cost_w,
                                              curr_row_weighted[j-1] + ins_cost_w,
                                              prev_row_weighted[j-1] + sub_cost_w });

            // Audio-based calculation
            float sub_cost_a = phon_dist_audio;
            float del_cost_a = GAP_BASE + GAP_PHONEME_SCALE * phon_dist_audio;
            float ins_cost_a = GAP_BASE + GAP_PHONEME_SCALE * phon_dist_audio;
            curr_row_audio[j] = std::min({ prev_row_audio[j] + del_cost_a,
                                           curr_row_audio[j-1] + ins_cost_a,
                                           prev_row_audio[j-1] + sub_cost_a });
        }
        
        std::swap(prev_row_weighted, curr_row_weighted);
        std::swap(prev_row_audio, curr_row_audio);
    }
    
    out_weighted_score = prev_row_weighted[len_b];
    out_audio_score = prev_row_audio[len_b];
}

std::vector<std::vector<std::string>> longest_contiguous_subsequence(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t len_a = a.size();
    size_t len_b = b.size();
    
    // Edge cases
    if (len_a == 0 || len_b == 0) {
        return std::vector<std::vector<std::string>>();
    }
    
    // Space-optimized: use only 2 rows instead of full 2D matrix
    // This reduces memory from O(n*m) to O(m), improving cache locality
    std::vector<int> prev_row(len_b + 1, 0);
    std::vector<int> curr_row(len_b + 1, 0);
    
    int max_len = 0;
    std::vector<std::pair<int, int>> ends;  // Store (i, j) positions where max subsequences end

    for (size_t i = 1; i <= len_a; ++i) {
        curr_row[0] = 0;
        
        for (size_t j = 1; j <= len_b; ++j) {
            if (a[i - 1] == b[j - 1]) {
                curr_row[j] = prev_row[j - 1] + 1;
                if (curr_row[j] > max_len) {
                    max_len = curr_row[j];
                    ends.clear();
                    ends.emplace_back(i, j);
                } else if (curr_row[j] == max_len) {
                    ends.emplace_back(i, j);
                }
            } else {
                curr_row[j] = 0;
            }
        }
        
        std::swap(prev_row, curr_row);
    }

    std::vector<std::vector<std::string>> result;
    for (const auto& [end_a, end_b] : ends) {
        if (max_len > 0) {
            result.emplace_back(a.begin() + end_a - max_len, a.begin() + end_a);
        }
    }
    return result;
}

std::vector<std::vector<std::string>> longest_contiguous_subsequence(const std::string& a, const std::string& b) {
    // OPTIMIZATION: Reserve space and use emplace_back to reduce allocations
    std::vector<std::string> vec_a;
    std::vector<std::string> vec_b;
    vec_a.reserve(a.size());
    vec_b.reserve(b.size());
    
    for (char c : a) vec_a.emplace_back(1, c);  // Construct string in-place
    for (char c : b) vec_b.emplace_back(1, c);
    
    return longest_contiguous_subsequence(vec_a, vec_b);
}

// Implementation of preload function declared in header
bool preload_phoneme_feature_distances(const std::string& db_path) {
    return preload_phoneme_distances_impl(db_path, "feature_based_distance", phoneme_distance_map);
}

bool preload_phoneme_audio_distances(const std::string& db_path) {
    return preload_phoneme_distances_impl(db_path, "audio_based_distance", phoneme_audio_distance_map);
}