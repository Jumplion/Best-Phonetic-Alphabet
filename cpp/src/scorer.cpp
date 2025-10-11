#include "../include/wordpair/scorer.h"
#include <algorithm>
#include <cctype>

// Very simple (placeholder) scoring: normalized Levenshtein-like distance on letters
static double normalized_letter_similarity(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    size_t match = 0;
    size_t len = std::min(a.size(), b.size());
    for (size_t i = 0; i < len; ++i) {
        if (std::tolower(a[i]) == std::tolower(b[i])) ++match;
    }
    double p = static_cast<double>(match) / static_cast<double>(std::max(a.size(), b.size()));
    return p;
}

ScoreResult score_word_pair(const std::string& a, const std::string& b) {
    ScoreResult r;
    r.word_a = a;
    r.word_b = b;
    double sim = normalized_letter_similarity(a, b);
    // simple transform to a "score"
    r.score = sim * 100.0;
    return r;
}
