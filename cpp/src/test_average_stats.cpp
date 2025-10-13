#include "../include/wordpair/db_writer.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    const std::string db_path = "f:/Repos/Best Phonetic Alphabet/data/BestPhonetics.db";
    
    // Parse command line arguments
    int test_n = -1; // Default to -1 (all words)
    if (argc > 1) {
        test_n = std::atoi(argv[1]);
    }

    std::cout << "=================================================\n";
    std::cout << "Average Statistics Computation Test\n";
    std::cout << "=================================================\n";
    
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

    // Limit to test_n words if specified
    if (test_n > 0 && all_words.size() > static_cast<size_t>(test_n)) {
        all_words.resize(test_n);
        std::cout << "Limited to first " << test_n << " words for testing.\n\n";
    } else if (test_n <= 0) {
        std::cout << "Computing statistics for ALL " << all_words.size() << " words in database.\n\n";
    }

    // Compute and write average statistics in batches
    // Batch size: use 1000 for full run, 500 for tests
    size_t batch_size = (test_n > 0 && test_n < 5000) ? 500 : 1000;
    std::cout << "Using batch size: " << batch_size << "\n\n";
    
    size_t total_written = db_writer.compute_and_write_average_stats_batched(all_words, batch_size, 5, 4);

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
