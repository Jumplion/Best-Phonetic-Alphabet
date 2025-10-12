#include <iostream>
#include <chrono>
#include <iomanip>
#include "../include/wordpair/scorer.h"
#include "../include/wordpair/db_writer.h"

int main(int argc, char** argv) {
    std::string db_path = "../data/BestPhonetics.db"; // default, relative to executable
    if (argc > 1) {
        db_path = argv[1];
    }
    DBWriter writer(db_path);
    if (!writer.init()) {
        std::cerr << "Failed to initialize DBWriter\n";
        return 1;
    }

    // Load all words from the database
    std::cout << "Loading words from database...\n";
    auto words = writer.load_all_words();
    
    if (words.empty()) {
        std::cerr << "No words loaded. Check database path and schema.\n";
        return 1;
    }

    std::cout << "✅ Successfully loaded " << words.size() << " words.\n\n";

    // Determine word count for testing (support --test-n=<count> flag)
    size_t num_words = words.size();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--test-n=") == 0) {
            size_t requested = std::stoul(arg.substr(9));
            num_words = std::min(requested, words.size());
            std::cout << "🧪 Test mode: processing first " << num_words << " words\n";
        }
    }

    // Calculate total number of pairs
    size_t total_pairs = (num_words * (num_words - 1)) / 2;
    std::cout << "Computing scores for " << total_pairs << " word pairs...\n\n";

    // Vector to store all computed scores
    std::vector<WordPairScore> scores;
    scores.reserve(total_pairs);

    // Progress tracking
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t pairs_computed = 0;
    size_t report_interval = std::max(size_t(1000), total_pairs / 100); // Report every 1% or 1000 pairs

    // Iterate through all unique pairs (i, j) where i < j
    for (size_t i = 0; i < num_words; ++i) {
        const Word& word_a = words[i];
        auto phonemes_a = word_a.get_phonemes();

        for (size_t j = i + 1; j < num_words; ++j) {
            const Word& word_b = words[j];
            auto phonemes_b = word_b.get_phonemes();

            // Compute scores for this pair
            WordPairScore score;
            score.word_id_1 = word_a.id;
            score.word_id_2 = word_b.id;

            // 1. Orthographic Levenshtein distance
            score.orth_levenshtein = orthographic_levenshtein_score(word_a.word, word_b.word);

            // 2. Phonetic Levenshtein distance
            score.phon_levenshtein = phonetic_levenshtein_score(phonemes_a, phonemes_b);

            // 3. Longest contiguous subsequence (orthographic)
            auto lcs_results = longest_contiguous_subsequence(word_a.word, word_b.word);
            if (!lcs_results.empty() && !lcs_results[0].empty()) {
                score.lcs_length = lcs_results[0].size();
                // Concatenate the first LCS result for storage
                score.lcs_text = "";
                for (const auto& s : lcs_results[0]) {
                    score.lcs_text += s;
                }
            } else {
                score.lcs_length = 0;
                score.lcs_text = "";
            }

            scores.push_back(score);
            pairs_computed++;

            // Progress reporting
            if (pairs_computed % report_interval == 0 || pairs_computed == total_pairs) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                double progress = 100.0 * pairs_computed / total_pairs;
                double pairs_per_sec = pairs_computed / std::max(1.0, static_cast<double>(elapsed.count()));
                size_t remaining = total_pairs - pairs_computed;
                double eta_sec = remaining / std::max(1.0, pairs_per_sec);

                std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "% "
                          << "(" << pairs_computed << "/" << total_pairs << ") | "
                          << "Speed: " << static_cast<size_t>(pairs_per_sec) << " pairs/s | "
                          << "ETA: " << static_cast<size_t>(eta_sec) << "s\n";
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n✅ Computed " << scores.size() << " word pair scores in " 
              << total_duration.count() << " seconds.\n";

    // Display sample results
    std::cout << "\nSample scored pairs:\n";
    for (size_t i = 0; i < std::min(size_t(5), scores.size()); ++i) {
        const auto& s = scores[i];
        std::cout << "  Pair [" << s.word_id_1 << ", " << s.word_id_2 << "]: "
                  << "orth_lev=" << s.orth_levenshtein << ", "
                  << "phon_lev=" << s.phon_levenshtein << ", "
                  << "lcs_len=" << s.lcs_length << " (\"" << s.lcs_text << "\")\n";
    }

    return 0;
}
