#pragma once

#include <string>
#include <vector>

struct ScoreResult {
    std::string word_a;
    std::string word_b;
    double score;
};

// Simple scoring API: compute a score for a pair of words
ScoreResult score_word_pair(const std::string& a, const std::string& b);

