#include "../include/wordpair/db_writer.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    const std::string db_path = "f:/Repos/Best Phonetic Alphabet/data/BestPhonetics.db";
    
    // Parse command line arguments
    int test_n = 100; // Default to 100 words
    if (argc > 1) {
        test_n = std::atoi(argv[1]);
    }

    std::cout << "=================================================\n";
    std::cout << "Average Statistics Computation Test\n";
    std::cout << "=================================================\n";
    std::cout << "Testing with first " << test_n << " words\n\n";

    // Initialize database
    DBWriter db_writer(db_path);
    if (!db_writer.init()) {
        std::cerr << "Failed to initialize database connection.\n";
        return 1;
    }

    // Load all words
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Word> all_words = db_writer.load_all_words();
    auto end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "⏱️  Loading took " << load_ms << " ms\n\n";

    if (all_words.empty()) {
        std::cerr << "No words loaded from database.\n";
        return 1;
    }

    // Limit to test_n words
    if (all_words.size() > static_cast<size_t>(test_n)) {
        all_words.resize(test_n);
        std::cout << "Limited to first " << test_n << " words for testing.\n\n";
    }

    // Compute average statistics
    std::cout << "Computing average statistics...\n";
    start = std::chrono::high_resolution_clock::now();
    auto stats = DBWriter::compute_average_stats(all_words, 5, 4);
    end = std::chrono::high_resolution_clock::now();
    auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "⏱️  Computation took " << compute_ms << " ms\n\n";

    // Display a few sample statistics
    std::cout << "\nSample Statistics (first 5 words):\n";
    std::cout << "-----------------------------------\n";
    for (size_t i = 0; i < std::min(size_t(5), stats.size()); ++i) {
        const auto& stat = stats[i];
        const auto& word = all_words[i];
        std::cout << "Word: " << word.word << " (ID: " << stat.word_id << ")\n";
        std::cout << "  Avg Orth: " << stat.avg_orth_levenshtein 
                  << "  Avg Phon: " << stat.avg_phon_levenshtein
                  << "  Avg LCS: " << stat.avg_lcs_length << "\n";
        std::cout << "  Close Orth matches: " << stat.count_close_orth
                  << "  Close Phon matches: " << stat.count_close_phon << "\n";
        std::cout << "  Orth range: [" << stat.min_orth_levenshtein 
                  << ", " << stat.max_orth_levenshtein << "]\n";
        std::cout << "  Phon range: [" << stat.min_phon_levenshtein 
                  << ", " << stat.max_phon_levenshtein << "]\n\n";
    }

    // Write to database
    std::cout << "Writing statistics to database...\n";
    start = std::chrono::high_resolution_clock::now();
    size_t written = db_writer.batch_write_average_stats(stats);
    end = std::chrono::high_resolution_clock::now();
    auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "✅ Wrote " << written << " statistics rows\n";
    std::cout << "⏱️  Writing took " << write_ms << " ms\n\n";

    // Test filtered loading with different thresholds
    std::cout << "\nTesting filtered word loading:\n";
    std::cout << "-----------------------------------\n";
    
    // Test 1: Very strict filter
    auto filtered1 = db_writer.load_filtered_words(10.0, 8.0, 10);
    std::cout << "Filter (avg_orth<=10, avg_phon<=8, close>=10): " 
              << filtered1.size() << " words\n";

    // Test 2: Moderate filter
    auto filtered2 = db_writer.load_filtered_words(15.0, 12.0, 5);
    std::cout << "Filter (avg_orth<=15, avg_phon<=12, close>=5): " 
              << filtered2.size() << " words\n";

    // Test 3: Loose filter
    auto filtered3 = db_writer.load_filtered_words(20.0, 15.0, 2);
    std::cout << "Filter (avg_orth<=20, avg_phon<=15, close>=2): " 
              << filtered3.size() << " words\n";

    std::cout << "\n=================================================\n";
    std::cout << "Test complete!\n";
    std::cout << "=================================================\n";

    return 0;
}
