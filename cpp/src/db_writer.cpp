#include "../include/wordpair/db_writer.h"
#include <sqlite3.h>
#include <iostream>
#include <sstream>

// ============================================================
// Word struct implementation
// ============================================================

std::vector<std::string> Word::get_phonemes() const {
    std::vector<std::string> phonemes;
    std::istringstream iss(normalized_phoneme_list);
    std::string phoneme;
    while (iss >> phoneme) {
        phonemes.push_back(phoneme);
    }
    return phonemes;
}

// ============================================================
// DBWriter implementation
// ============================================================

struct DBWriter::Impl {
    sqlite3* db = nullptr;
};

DBWriter::DBWriter(const std::string& db_path) : db_path_(db_path), impl_(new Impl()) {}

DBWriter::~DBWriter() {
    if (impl_) {
        if (impl_->db) {
            sqlite3_close(impl_->db);
        }
        delete impl_;
    }
}

bool DBWriter::init() {
    int rc = sqlite3_open(db_path_.c_str(), &impl_->db);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open DB: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }
    return true;
}

std::vector<Word> DBWriter::load_all_words() const {
    std::vector<Word> words;
    
    if (!impl_ || !impl_->db) {
        std::cerr << "DBWriter not initialized. Call init() first.\n";
        return words;
    }

    const char* sql = "SELECT id, word, normalized_phoneme_list FROM words ORDER BY id;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare SQL: " << sqlite3_errmsg(impl_->db) << std::endl;
        return words;
    }

    // Reserve space for typical CMU dictionary size (~130k words)
    words.reserve(150000);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        Word w;
        w.id = sqlite3_column_int(stmt, 0);
        
        const unsigned char* word_text = sqlite3_column_text(stmt, 1);
        if (word_text) {
            w.word = reinterpret_cast<const char*>(word_text);
        }
        
        const unsigned char* phoneme_text = sqlite3_column_text(stmt, 2);
        if (phoneme_text) {
            w.normalized_phoneme_list = reinterpret_cast<const char*>(phoneme_text);
        }
        
        words.push_back(std::move(w));
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Error reading words: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    
    std::cout << "✅ Loaded " << words.size() << " words from database.\n";
    return words;
}