#pragma once

#include <string>
#include <vector>

/**
 * Compute an orthographic similarity score between two strings using the Levenshtein edit distance.
 *
 * The score is defined as the minimum number of single-character edits (insertions, deletions, or substitutions)
 * required to change one string into the other.
 *
 * @param a First input string.
 * @param b Second input string.
 * @return Integer Levenshtein distance between the two strings.
 *
 * @note 
 * - Comparison is case-insensitive. Inputs will be normalized to lowercase.
 * @note
 * - Ties to orthography: this measures character-level edit similarity (insertions, deletions, substitutions).
 */
int orthographic_levenshtein_score(const std::string& a, const std::string& b);

/**
 * Compute a phonetic similarity score between two sequences of phonemes using the Levenshtein edit distance.
 *
 * The score is defined as the minimum number of single-phoneme edits (insertions, deletions, or substitutions)
 * required to change one phoneme sequence into the other.
 *
 * @param a First input vector of phonemes (strings).
 * @param b Second input vector of phonemes (strings).
 * @return Integer Levenshtein distance between the two phoneme sequences.
 *
 * @note 
 * - The function operates on vectors of strings representing phonemes.
 */
int phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b);

/**
 * Compute a weighted phonetic similarity score between two sequences of phonemes using a weighted Levenshtein edit distance.
 * 
 * The score is defined as the minimum weighted number of single-phoneme edits (insertions, deletions, or substitutions)
 * required to change one phoneme sequence into the other. Weights can be assigned based on phonetic similarity.
 * 
 * In this version, different types of edits can have different costs depending on phonetic similarity and phonetic features between phonemes.
 * Phonetic features from the database are used to construct the "distance" between phonemes that will be used for weighting.
 * 
 * @param a First input vector of phonemes (strings).
 * @param b Second input vector of phonemes (strings).
 * @return Float weighted Levenshtein distance between the two phoneme sequences.
 *
 * @note 
 * - The function operates on vectors of strings representing phonemes.
 *
 * - Phonetic features and similarity metrics should be defined to determine the weights for different edits.
 *
 * - Example: Substituting a vowel (AH) for another vowel (OH) might have a lower cost than substituting a vowel (AH) for a consonant (K).
 */
float weighted_phonetic_levenshtein_score(const std::vector<std::string>& a, const std::vector<std::string>& b);

/**
 * Preload phoneme feature-based distances from a SQLite database.
 *
 * The function reads the `phoneme_distances` table and the `feature_based_distance`
 * column and stores symmetric distances for fast lookup. Call this once at startup
 * (before calling weighted_phonetic_levenshtein_score) to use database distances.
 *
 * @param db_path Path to the SQLite database (e.g., "data/BestPhonetics.db").
 * @return true on success, false on failure.
 */
bool preload_phoneme_feature_distances(const std::string& db_path);

/**
 * Compute the Longest Contiguous Subsequence (LCS) of two sequences of tokens.
 *
 * Given two sequences of strings 'a' and 'b', find the longest contiguous sequence
 * of elements that appears in both 'a' and 'b'.
 *
 * The returned vector contains the longest contiguous subsequence. If multiple distinct
 * subsequences of the same maximum length exist, all are returned.
 *
 * Behavior and edge cases:
 * 
 * - If either input is empty, an empty vector is returned.
 * 
 * - Input sequences are not modified.
 * 
 * - Equality comparisons are case-sensitive unless the caller normalizes inputs.
 *
 * Example 1:
 * 
 * - a = "aaaabbbbcccc"
 * 
 * - b = "xxbbbbyyy"
 * 
 * - Result = [["b", "b", "b", "b"]]
 * 
 * Example 2:
 * 
 * - a = "abcde"
 * 
 * - b = "abxyzde"
 * 
 * - Result = [["a", "b"], ["d", "e"]]
 *
 * @param a First string.
 * @param b Second string.
 * @return A vector containing all the longest contiguous subsequence elements. Usually only one such subsequence exists, however if multiple distinct subsequences of the same maximum length exist then all are returned.
 */
std::vector<std::vector<std::string>> longest_contiguous_subsequence(const std::string& a, const std::string& b);

/**
 * Compute the Longest Contiguous Subsequence (LCS) of two sequences of phonemes.
 *
 * Given two sequences of strings/phonemes 'a' and 'b', find the longest contiguous sequence
 * of elements that appears in both 'a' and 'b'.
 *
 * The returned vector contains the longest contiguous subsequence. If multiple distinct
 * subsequences of the same maximum length exist, all are returned.
 *
 * Behavior and edge cases:
 * 
 * - If either input is empty, an empty vector is returned.
 * 
 * - Input sequences are not modified.
 * 
 * - Equality comparisons are case-sensitive unless the caller normalizes inputs.
 *
 * Example 1:
 * 
 * - a = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c"]
 * 
 * - b = ["x", "x", "b", "b", "b", "b", "y", "y", "y"]
 * 
 * - Result = ["b", "b", "b", "b"]
 * 
 * Example 2:
 * 
 * - a = ["a", "b", "c", "d", "e"]
 * 
 * - b = ["a", "b", "x", "y", "z", "d", "e"]
 * 
 * - Result = [["a", "b"], ["d", "e"]]
 *
 * @param a First vector of strings.
 * @param b Second vector of strings.
 * @return A vector containing all the longest contiguous subsequence elements. Usually only one such subsequence exists, however if multiple distinct subsequences of the same maximum length exist then all are returned.
 */
std::vector<std::vector<std::string>> longest_contiguous_subsequence(const std::vector<std::string>& a, const std::vector<std::string>& b);

/**
 * Compute the orthographic Jaccard index between two strings.
 *
 * The orthographic Jaccard index is defined as the size of the intersection
 * divided by the size of the union of the sets of orthographic units
 * extracted from each input string. By default, orthographic units are the
 * unique characters (Unicode code points) present in each string.
 *
 * The return value is a floating-point similarity in the range [0.0, 1.0]:
 *   - 1.0 means the orthographic sets are identical
 *   - 0.0 means there is no overlap between the orthographic sets
 *
 * @note
 *   - Comparison is case-sensitive: characters with different case are treated
 *     as distinct units.
 * 
 *   - If both input strings contain no orthographic units (e.g., both are
 *     empty), the function treats the sets as identical and returns 1.0.
 *
 * @param a   First input string.
 * @param b   Second input string.
 * @return    Jaccard index (|intersection| / |union|) as a float in [0,1].
 */
float orthographic_jaccard_index(const std::string& a, const std::string& b);

/**
 * Compute the phonetic Jaccard index between two sequences of phonemes.
 *
 * The phonetic Jaccard index is defined as the size of the intersection
 * divided by the size of the union of the sets of phonemes extracted from
 * each input sequence.
 *
 * The return value is a floating-point similarity in the range [0.0, 1.0]:
 *   - 1.0 means the phoneme sets are identical
 *   - 0.0 means there is no overlap between the phoneme sets
 *
 * @note
 *   - If both input sequences contain no phonemes (e.g., both are empty),
 *     the function treats the sets as identical and returns 1.0.
 *
 * @param a   First input vector of phonemes (strings).
 * @param b   Second input vector of phonemes (strings).
 * @return    Jaccard index (|intersection| / |union|) as a float in [0,1].
 */
float phonetic_jaccard_index(const std::vector<std::string>& a, const std::vector<std::string>& b);
