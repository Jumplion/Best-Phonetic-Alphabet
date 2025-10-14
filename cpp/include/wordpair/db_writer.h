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
    std::string word_1;  // Actual word strings for database storage
    std::string word_2;
    
    // Basic distance metrics
    int orth_levenshtein;
    int phon_levenshtein;
    
    // Longest contiguous subsequence (we'll store the length)
    int lcs_length;
    std::string lcs_text;  // For orthographic LCS, store the actual sequence
};

// Represents average statistics for a single word
struct WordAverageStats {
    int word_id;
    
    // Average distances
    double avg_orth_levenshtein;
    double avg_phon_levenshtein;
    double avg_lcs_length;
    
    // Distribution statistics
    int min_orth_levenshtein;
    int max_orth_levenshtein;
    double stddev_orth_levenshtein;
    
    int min_phon_levenshtein;
    int max_phon_levenshtein;
    double stddev_phon_levenshtein;
    
    // Count of close matches
    int count_close_orth;  // orth_lev <= 5
    int count_close_phon;  // phon_lev <= 4
};

// Represents statistics for a specific metric (e.g., weighted phonetic Levenshtein)
struct WordMetricStats {
    int word_id;
    double avg_value;
    double min_value;
    double max_value;
    double stddev_value;
    int count_close;  // count below a threshold
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

    // Write a batch of word pair scores to the database
    // Uses a single transaction for efficiency
    // Returns number of rows successfully written
    size_t batch_write_scores(const std::vector<WordPairScore>& scores);

    // Write average statistics for words to the database
    // Returns number of rows successfully written
    size_t batch_write_average_stats(const std::vector<WordAverageStats>& stats);

    // Load words filtered by average stats criteria
    // Returns words that meet the threshold requirements
    std::vector<Word> load_filtered_words(
        double max_avg_orth_lev = 20.0,
        double max_avg_phon_lev = 15.0,
        int min_close_matches = 5
    ) const;

    // Compute and write average statistics in batches
    // Processes words in chunks, writing results as we go to prevent memory overflow
    // Returns total number of statistics rows written
    // batch_size: number of words to process before writing to database
    size_t compute_and_write_average_stats_batched(
        const std::vector<Word>& words,
        size_t batch_size = 1000,
        int orth_close_threshold = 5,
        int phon_close_threshold = 4
    );

    // Compute and update a specific metric (e.g., weighted phonetic Levenshtein) for all words
    // Updates existing average_stats rows with the specified column
    // column_name: the database column to update (e.g., "avg_weighted_phon_lev")
    // score_function: function that takes two phoneme vectors and returns a score
    // close_threshold: threshold for counting "close" matches
    // Returns total number of rows updated
    size_t compute_and_update_metric_batched(
        const std::vector<Word>& words,
        const std::string& column_prefix,  // e.g., "weighted_phon_lev" for avg_, min_, max_, stddev_
        float (*score_function)(const std::vector<std::string>&, const std::vector<std::string>&),
        size_t batch_size = 1000,
        float close_threshold = 5.0f
    );

    // Helper: Update specific metric columns in average_stats table
    // Uses UPDATE to modify existing rows based on word_id
    size_t batch_update_metric_stats(
        const std::vector<WordMetricStats>& stats,
        const std::string& column_prefix
    );

private:
    std::string db_path_;
    struct Impl;
    Impl* impl_ = nullptr;

};
