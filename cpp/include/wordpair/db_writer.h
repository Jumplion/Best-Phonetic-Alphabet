#pragma once

#include <string>
#include "scorer.h"

class DBWriter {
public:
    explicit DBWriter(const std::string& db_path);
    ~DBWriter();

    // Initialize DB (create table if not exists)
    bool init();

    // Insert a score result
    bool insert_score(const ScoreResult& res);

private:
    std::string db_path_;
    struct Impl;
    Impl* impl_ = nullptr;
};
