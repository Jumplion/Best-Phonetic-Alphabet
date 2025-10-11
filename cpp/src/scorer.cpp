#include "../include/wordpair/scorer.h"
#include <algorithm>
#include <cctype>

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
    // Placeholder implementation: same as unweighted for now
    return static_cast<float>(phonetic_levenshtein_score(a, b));
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
    std::vector<std::string> vec_a(a.begin(), a.end());
    std::vector<std::string> vec_b(b.begin(), b.end());
    return longest_contiguous_subsequence(vec_a, vec_b);
}