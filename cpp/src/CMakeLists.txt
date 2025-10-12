add_executable(wordpair_scorer
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scorer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/db_writer.cpp
)

target_include_directories(wordpair_scorer PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link SQLite
if (TARGET SQLite::SQLite3)
    target_link_libraries(wordpair_scorer PRIVATE SQLite::SQLite3)
else()
    if (DEFINED SQLite3_LIBRARIES)
        target_link_libraries(wordpair_scorer PRIVATE ${SQLite3_LIBRARIES})
    else()
        message(FATAL_ERROR "SQLite library not found. Please set SQLITE3_LIBRARY and SQLITE3_INCLUDE_DIR or install SQLite and retry.")
    endif()
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(wordpair_scorer PRIVATE OpenMP::OpenMP_CXX)
endif()

# install target (optional)
install(TARGETS wordpair_scorer RUNTIME DESTINATION bin)
