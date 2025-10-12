#include <iostream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
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
    std::cout << "Computing scores for " << total_pairs << " word pairs...\n";

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "🚀 Parallel mode: using " << num_threads << " threads\n\n";
#else
    std::cout << "⚠️  Single-threaded mode (OpenMP not available)\n\n";
#endif

    // Pre-allocate vector to store all computed scores
    std::vector<WordPairScore> scores(total_pairs);

    // Progress tracking
    auto start_time = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> pairs_computed{0};
    size_t report_interval = std::max(size_t(10000), total_pairs / 100); // Report every 1% or 10000 pairs
    std::mutex progress_mutex;

    // Parallel computation of all pairs
    // We use a single parallelized loop over a flat index, then map to (i,j) pairs
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
        // Map linear index to (i, j) pair where i < j
        // For pair_idx, find i and j such that pair_idx = i*(2*n - i - 1)/2 + (j - i - 1)
        // Using inverse formula: i = n - 2 - floor(sqrt(-8*pair_idx + 4*n*(n-1) - 7)/2 - 0.5)
        size_t i = 0;
        size_t remaining = pair_idx;
        size_t pairs_for_i = num_words - 1;
        
        // Find which 'i' this pair_idx belongs to
        while (remaining >= pairs_for_i) {
            remaining -= pairs_for_i;
            i++;
            pairs_for_i--;
        }
        size_t j = i + 1 + remaining;

        const Word& word_a = words[i];
        const Word& word_b = words[j];
        
        // Parse phonemes (each thread gets its own copy)
        auto phonemes_a = word_a.get_phonemes();
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

        // Store result in pre-allocated vector (thread-safe since each thread writes to unique index)
        scores[pair_idx] = score;

        // Atomic increment for progress tracking
        size_t current_count = ++pairs_computed;

        // Progress reporting (throttled with mutex to avoid spam)
        if (current_count % report_interval == 0 || current_count == total_pairs) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            double progress = 100.0 * current_count / total_pairs;
            double pairs_per_sec = current_count / std::max(1.0, static_cast<double>(elapsed.count()));
            size_t remaining_pairs = total_pairs - current_count;
            double eta_sec = remaining_pairs / std::max(1.0, pairs_per_sec);
            
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "% "
                << "(" << current_count << "/" << total_pairs << ") | "
                << "Speed: " << static_cast<size_t>(pairs_per_sec) << " pairs/s | "
                << "ETA: " << static_cast<size_t>(eta_sec) << "s" << std::flush;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n\n✅ Computed " << scores.size() << " word pair scores in " 
              << total_duration.count() << " seconds";
    
    if (total_duration.count() > 0) {
        size_t avg_speed = scores.size() / total_duration.count();
        std::cout << " (avg " << avg_speed << " pairs/s)";
    }
    std::cout << ".\n";

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
