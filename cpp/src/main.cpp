#include <iostream>
#include "../include/wordpair/scorer.h"
#include "../include/wordpair/db_writer.h"

int main(int argc, char** argv) {
    std::string db_path = "phoneme_data.db"; // default, relative to executable
    if (argc > 1) db_path = argv[1];

    DBWriter writer(db_path);
    if (!writer.init()) {
        std::cerr << "Failed to initialize DBWriter\n";
        return 1;
    }

    // Demo scoring
    std::string word_a = "example";
    std::string word_b = "samples";
    float score = orthographic_jaccard_index(word_a, word_b);
    if (!writer.insert_score(word_a, word_b, score)) {
        std::cerr << "Failed to insert score\n";
        return 1;
    }

    std::cout << "Inserted score for " << word_a << " - " << word_b << " = " << score << "\n";
    return 0;
}
