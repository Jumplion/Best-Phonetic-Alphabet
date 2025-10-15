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

    // Determine batch size (support --batch-size=N flag)
    size_t batch_size = 50000;  // Default: write every 50K pairs
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--batch-size=") == 0) {
            batch_size = std::stoul(arg.substr(13));
            std::cout << "📦 Batch size set to: " << batch_size << " pairs\n";
        }
    }

    // Calculate total number of pairs
    size_t total_pairs = (num_words * (num_words - 1)) / 2;
    std::cout << "Computing scores for " << total_pairs << " word pairs...\n";

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "🚀 Parallel mode: using " << num_threads << " threads\n";
#else
    std::cout << "⚠️  Single-threaded mode (OpenMP not available)\n";
#endif
    std::cout << "💾 Writing to database in batches of " << batch_size << " pairs\n\n";

    // Pre-parse all phonemes once before parallel computation (Priority 4 optimization)
    std::cout << "📝 Pre-parsing phonemes for all " << num_words << " words...\n";
    std::vector<std::vector<std::string>> all_phonemes(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        all_phonemes[i] = words[i].get_phonemes();
    }
    std::cout << "✅ Phonemes pre-parsed (" << num_words << " words)\n\n";

    // Batch buffer for accumulating scores before writing
    std::vector<WordPairScore> batch_buffer;
    batch_buffer.reserve(batch_size);
    std::mutex batch_mutex;
    std::mutex db_write_mutex;  // Serialize database writes

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
        
        // Use pre-parsed phonemes instead of parsing per-pair (Priority 4 optimization)
        const auto& phonemes_a = all_phonemes[i];
        const auto& phonemes_b = all_phonemes[j];

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

        // IDs are already set (word_id_1, word_id_2) - no need for text fields

        // Add score to batch buffer (critical section)
        bool should_write = false;
        std::vector<WordPairScore> write_batch;
        {
            std::lock_guard<std::mutex> lock(batch_mutex);
            batch_buffer.push_back(std::move(score));
            
            // If batch is full, prepare to write
            if (batch_buffer.size() >= batch_size) {
                write_batch = std::move(batch_buffer);
                batch_buffer.clear();
                batch_buffer.reserve(batch_size);
                should_write = true;
            }
        }

        // Write batch outside critical section (serialize with mutex)
        if (should_write) {
            std::lock_guard<std::mutex> write_lock(db_write_mutex);
            size_t written = writer.batch_write_scores(write_batch);
            if (written != write_batch.size()) {
                std::cerr << "\n⚠️  Warning: Only wrote " << written << "/" << write_batch.size() << " scores\n";
            }
        }

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

    // Write any remaining scores in the buffer
    if (!batch_buffer.empty()) {
        std::cout << "\n💾 Writing final batch of " << batch_buffer.size() << " pairs...\n";
        std::lock_guard<std::mutex> write_lock(db_write_mutex);
        size_t written = writer.batch_write_scores(batch_buffer);
        if (written != batch_buffer.size()) {
            std::cerr << "⚠️  Warning: Only wrote " << written << "/" << batch_buffer.size() << " scores\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n✅ Computed and wrote " << pairs_computed.load() << " word pair scores in " 
              << total_duration.count() << " seconds";
    
    if (total_duration.count() > 0) {
        size_t avg_speed = pairs_computed.load() / total_duration.count();
        std::cout << " (avg " << avg_speed << " pairs/s)";
    }
    std::cout << ".\n";

    return 0;
}
