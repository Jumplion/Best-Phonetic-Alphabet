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
    auto r = score_word_pair("alpha", "alfa");
    if (!writer.insert_score(r)) {
        std::cerr << "Failed to insert score\n";
        return 1;
    }

    std::cout << "Inserted score for " << r.word_a << " - " << r.word_b << " = " << r.score << "\n";
    return 0;
}
