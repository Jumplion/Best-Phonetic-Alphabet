#include "../include/wordpair/scorer.h"
#include <algorithm>
#include <cctype>
#include <sqlite3.h>
#include <unordered_map>
#include <mutex>
#include <iostream>

// File-scope phoneme distance caches
static std::unordered_map<std::string, float> phoneme_distance_map;  // feature-based distances
static std::unordered_map<std::string, float> phoneme_audio_distance_map;  // audio-based distances
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

// Helper: generic weighted phonetic Levenshtein using specified distance map
static float weighted_phonetic_levenshtein_impl(
    const std::vector<std::string>& a, 
    const std::vector<std::string>& b,
    const std::unordered_map<std::string, float>& distance_map,
    const char* map_name
) {
    // phoneme distance lookup
    auto map_lookup = [&](const std::string &p1, const std::string &p2, float &out_val) -> bool {
        std::string k1 = p1 + "|" + p2;
        std::lock_guard<std::mutex> lk(phoneme_map_mutex);
        auto it = distance_map.find(k1);
        if (it != distance_map.end()) { out_val = it->second; return true; }
        // Check reverse key
        std::string k2 = p2 + "|" + p1;
        it = distance_map.find(k2);
        if (it != distance_map.end()) { out_val = it->second; return true; }
        return false;
    };

    auto phoneme_distance = [&](const std::string &p1, const std::string &p2) -> float {
        if (p1 == p2) return 0.0f;
        float val = 0.0f;
        if (map_lookup(p1, p2, val)) return val;

        throw std::runtime_error(
            std::string("Missing phoneme distance entry in ") + map_name + " for phoneme pair ('" + 
            p1 + "', '" + p2 + "'). Please ensure the appropriate preload function was called."
        );
    };

    const float GAP_BASE = 0.6f;
    const float GAP_PHONEME_SCALE = 0.8f;

    size_t len_a = a.size();
    size_t len_b = b.size();
    
    if (len_a == 0) return len_b * GAP_BASE;
    if (len_b == 0) return len_a * GAP_BASE;

    std::vector<float> prev_row(len_b + 1);
    std::vector<float> curr_row(len_b + 1);

    prev_row[0] = 0.0f;
    for (size_t j = 1; j <= len_b; ++j) {
        prev_row[j] = prev_row[j-1] + GAP_BASE;
    }

    for (size_t i = 1; i <= len_a; ++i) {
        curr_row[0] = prev_row[0] + GAP_BASE;
        
        for (size_t j = 1; j <= len_b; ++j) {
            float sub_cost = phoneme_distance(a[i-1], b[j-1]);
            float del_cost = GAP_BASE + GAP_PHONEME_SCALE * phoneme_distance(a[i-1], b[j-1]);
            float ins_cost = GAP_BASE + GAP_PHONEME_SCALE * phoneme_distance(a[i-1], b[j-1]);

            curr_row[j] = std::min({ prev_row[j] + del_cost,
                                     curr_row[j-1] + ins_cost,
                                     prev_row[j-1] + sub_cost });
        }
        
        std::swap(prev_row, curr_row);
    }
    
    return prev_row[len_b];
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
    std::vector<std::vector<int>> dp(len_a + 1, std::vector<int>(len_b + 1));

    for (size_t i = 0; i <= len_a; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= len_b; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= len_a; ++i) {
        for (size_t j = 1; j <= len_b; ++j) {
            int cost = (a_lower[i - 1] == b_lower[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({ dp[i - 1][j] + 1,      // Deletion
                                  dp[i][j - 1] + 1,      // Insertion
                                  dp[i - 1][j - 1] + cost }); // Substitution
        }
    }
    return dp[len_a][len_b];
}

int phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t len_a = a.size();
    size_t len_b = b.size();
    std::vector<std::vector<int>> dp(len_a + 1, std::vector<int>(len_b + 1));

    for (size_t i = 0; i <= len_a; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= len_b; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= len_a; ++i) {
        for (size_t j = 1; j <= len_b; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({ dp[i - 1][j] + 1,      // Deletion
                                  dp[i][j - 1] + 1,      // Insertion
                                  dp[i - 1][j - 1] + cost }); // Substitution
        }
    }
    return dp[len_a][len_b];
}

float weighted_phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    validate_normalized_phonemes(a, "a");
    validate_normalized_phonemes(b, "b");
    return weighted_phonetic_levenshtein_impl(a, b, phoneme_distance_map, "feature-based distance map");
}

float audio_phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    validate_normalized_phonemes(a, "a");
    validate_normalized_phonemes(b, "b");
    return weighted_phonetic_levenshtein_impl(a, b, phoneme_audio_distance_map, "audio-based distance map");
}

std::vector<std::vector<std::string>> longest_contiguous_subsequence(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t len_a = a.size();
    size_t len_b = b.size();
    std::vector<std::vector<int>> dp(len_a + 1, std::vector<int>(len_b + 1, 0));
    int max_len = 0;
    std::vector<std::pair<int, int>> ends;

    for (size_t i = 1; i <= len_a; ++i) {
        for (size_t j = 1; j <= len_b; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > max_len) {
                    max_len = dp[i][j];
                    ends.clear();
                    ends.emplace_back(i, j);
                } else if (dp[i][j] == max_len) {
                    ends.emplace_back(i, j);
                }
            }
        }
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
    std::vector<std::string> vec_a;
    std::vector<std::string> vec_b;
    for (char c : a) vec_a.push_back(std::string(1, c));
    for (char c : b) vec_b.push_back(std::string(1, c));
    return longest_contiguous_subsequence(vec_a, vec_b);
}

// Implementation of preload function declared in header
bool preload_phoneme_feature_distances(const std::string& db_path) {
    return preload_phoneme_distances_impl(db_path, "feature_based_distance", phoneme_distance_map);
}

bool preload_phoneme_audio_distances(const std::string& db_path) {
    return preload_phoneme_distances_impl(db_path, "audio_based_distance", phoneme_audio_distance_map);
}