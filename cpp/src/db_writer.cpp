#include "../include/wordpair/db_writer.h"
#include <sqlite3.h>
#include <iostream>
#include <sstream>

struct DBWriter::Impl {
    sqlite3* db = nullptr;
};

DBWriter::DBWriter(const std::string& db_path) : db_path_(db_path), impl_(new Impl()) {}

DBWriter::~DBWriter() {
    if (impl_) {
        if (impl_->db) sqlite3_close(impl_->db);
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

bool DBWriter::insert_score(const std::string& a, const std::string& b, float score) {
    const char* insert_sql = "INSERT INTO word_pair_scores (word_a, word_b, score) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, insert_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }
    sqlite3_bind_text(stmt, 1, a.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, b.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 3, score);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to execute insert: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_finalize(stmt);
        return false;
    }
    sqlite3_finalize(stmt);
    return true;
}
