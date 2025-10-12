#pragma once

#include <string>
#include <vector>
#include "scorer.h"

// Represents a single word entry from the database
struct Word {
    int id;
    std::string word;
    std::string normalized_phoneme_list; // space-separated phonemes like "B AH N AE N AH"
    
    // Parse phonemes into a vector (splits on whitespace)
    std::vector<std::string> get_phonemes() const;
};

// Represents computed scores for a word pair
struct WordPairScore {
    int word_id_1;
    int word_id_2;
    
    // Basic distance metrics
    int orth_levenshtein;
    int phon_levenshtein;
    
    // Longest contiguous subsequence (we'll store the length)
    int lcs_length;
    std::string lcs_text;  // For orthographic LCS, store the actual sequence
};

class DBWriter {
public:
    explicit DBWriter(const std::string& db_path);
    ~DBWriter();

    // Initialize DB (create table if not exists)
    bool init();

    // Load all words from the 'words' table into memory
    // Returns empty vector on error (check init() first)
    std::vector<Word> load_all_words() const;

private:
    std::string db_path_;
    struct Impl;
    Impl* impl_ = nullptr;

};
