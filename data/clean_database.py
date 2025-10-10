import sqlite3
import argparse
import shutil
from pathlib import Path

import tqdm


def backup_db(db_path: Path):
    backup_path = db_path.with_suffix(db_path.suffix + '.bak')
    shutil.copy2(db_path, backup_path)
    print(f"Backup created at {backup_path}")
    return backup_path


def remove_multiword_entries(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, word FROM words WHERE instr(word, ' ') > 0")
    rows = cur.fetchall()
    print(f"Found {len(rows)} multi-word entries (contain spaces)")
    ids = [r[0] for r in rows]
    if ids:
        cur.execute(f"DELETE FROM words WHERE id IN ({','.join(['?']*len(ids))})", ids)
        conn.commit()
    return len(ids)


def deduplicate_words(conn):
    cur = conn.cursor()
    # Count total rows and distinct groups to compute duplicate count
    cur.execute("SELECT COUNT(*) FROM words")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM (SELECT lower(word) as lw, normalized_phoneme_list FROM words GROUP BY lw, normalized_phoneme_list)")
    distinct_groups = cur.fetchone()[0]
    total_dups = max(0, total_rows - distinct_groups)

    # If there are no duplicates, return early
    if total_dups == 0:
        return 0, 0

    # Use a temp table to hold the ids we want to keep (min id per group), then delete others
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS keep_ids(id INTEGER PRIMARY KEY)")
    cur.execute("DELETE FROM keep_ids")
    cur.execute("INSERT INTO keep_ids(id) SELECT MIN(id) FROM words GROUP BY lower(word), normalized_phoneme_list")

    # Delete all rows whose id is not in keep_ids
    cur.execute("DELETE FROM words WHERE id NOT IN (SELECT id FROM keep_ids)")
    # The number of removed rows should equal total_dups, but compute from counts to be safe
    cur.execute("SELECT COUNT(*) FROM words")
    remaining = cur.fetchone()[0]
    removed = total_rows - remaining

    # Cleanup temp table
    cur.execute("DROP TABLE IF EXISTS keep_ids")
    conn.commit()
    return total_dups, removed


def remove_empty_pronunciations(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, word FROM words WHERE phoneme_list IS NULL OR trim(phoneme_list) = ''")
    rows = cur.fetchall()
    print(f"Found {len(rows)} entries with empty phoneme_list")
    batch = []
    removed = 0
    BATCH = 500
    for r in rows:
        batch.append(r[0])
        if len(batch) >= BATCH:
            cur.execute(f"DELETE FROM words WHERE id IN ({','.join(['?']*len(batch))})", batch)
            removed += len(batch)
            batch.clear()
    if batch:
        cur.execute(f"DELETE FROM words WHERE id IN ({','.join(['?']*len(batch))})", batch)
        removed += len(batch)
    if removed:
        conn.commit()
    return removed


def remove_non_alpha_words(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, word FROM words WHERE word GLOB '*[^A-Za-z-]+'")
    rows = cur.fetchall()
    print(f"Found {len(rows)} non-alphabetic word entries (allowing hyphens)")
    batch = []
    removed = 0
    BATCH = 500
    for r in rows:
        batch.append(r[0])
        if len(batch) >= BATCH:
            cur.execute(f"DELETE FROM words WHERE id IN ({','.join(['?']*len(batch))})", batch)
            removed += len(batch)
            batch.clear()
    if batch:
        cur.execute(f"DELETE FROM words WHERE id IN ({','.join(['?']*len(batch))})", batch)
        removed += len(batch)
    if removed:
        conn.commit()
    return removed


def lowercase_all_words(conn):
    cur = conn.cursor()
    # Update word column to be lowercased for all rows
    cur.execute("UPDATE words SET word = lower(word) WHERE word IS NOT NULL")
    affected = cur.rowcount if hasattr(cur, 'rowcount') else None
    conn.commit()
    print(f"Lowercased words in table (rows affected: {affected})")
    return affected


def remove_single_char_entries(conn):
    cur = conn.cursor()
    # Count single-character words (after trimming)
    cur.execute("SELECT COUNT(*) FROM words WHERE word IS NOT NULL AND LENGTH(TRIM(word)) = 1")
    cnt = cur.fetchone()[0]
    if cnt == 0:
        print("Found 0 single-character entries")
        return 0
    # Delete them in one statement
    cur.execute("DELETE FROM words WHERE word IS NOT NULL AND LENGTH(TRIM(word)) = 1")
    conn.commit()
    print(f"Removed {cnt} single-character entries")
    return cnt


def remove_duplicates_by_normalized_phonemes(conn):
    """For rows that share the same normalized_phoneme_list, keep the row
    whose word is shortest. If multiple rows have the same shortest length,
    keep the one with the smallest id. Delete the others.
    """
    cur = conn.cursor()
    # Create a temp table to hold ids to keep
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS keep_ids(id INTEGER PRIMARY KEY)")
    cur.execute("DELETE FROM keep_ids")

    # Create an index to speed grouping/joins (no-op if exists)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_words_norm_phoneme ON words(normalized_phoneme_list)")

    # Use a single SQL statement to compute the keeper id per normalized_phoneme_list:
    #  - For each normalized_phoneme_list, find the minimal LENGTH(word)
    #  - Among rows having that length, select the minimal id (tie-break)
    # Insert those ids into keep_ids in one set-based operation (fast)
    cur.execute("""
        INSERT INTO keep_ids(id)
        SELECT MIN(w.id) FROM words w
        JOIN (
            SELECT normalized_phoneme_list as npl, MIN(LENGTH(word)) as min_len
            FROM words
            GROUP BY npl
        ) g ON (
            (w.normalized_phoneme_list = g.npl OR (w.normalized_phoneme_list IS NULL AND g.npl IS NULL))
            AND LENGTH(w.word) = g.min_len
        )
        GROUP BY w.normalized_phoneme_list
    """)

    # Delete rows not in keep_ids
    cur.execute("SELECT COUNT(*) FROM words")
    total_before = cur.fetchone()[0]

    cur.execute("DELETE FROM words WHERE id NOT IN (SELECT id FROM keep_ids)")

    cur.execute("SELECT COUNT(*) FROM words")
    total_after = cur.fetchone()[0]
    removed = total_before - total_after

    cur.execute("DROP TABLE IF EXISTS keep_ids")
    conn.commit()
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db', nargs='?', default='BestPhonetics.db')
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    backup_db(db_path)

    conn = sqlite3.connect(db_path)
    #print("Removing multi-word entries...")
    #removed_multi = remove_multiword_entries(conn)
    #print(f"Removed multi-word entries: {removed_multi}")
#
    #print("Deduplicating words by (lower(word), normalized_phoneme_list)...")
    #total_dups, total_removed = deduplicate_words(conn)
    #print(f"Found duplicate entries (extra rows): {total_dups}")
    #print(f"Removed duplicates: {total_removed}")
#
    #print("Removing entries with empty phoneme_list...")
    #removed_empty_pron = remove_empty_pronunciations(conn)
    #print(f"Removed entries with empty phoneme_list: {removed_empty_pron}")
#
    #print("Removing non-alphabetic word entries (allowing hyphens)...")
    #removed_non_alpha = remove_non_alpha_words(conn)
    #print(f"Removed non-alphabetic word entries: {removed_non_alpha}")

    #print("Removing single-character entries...")
    #removed_single = remove_single_char_entries(conn)
    #print(f"Removed single-character entries: {removed_single}")

    #print("Making all words lowercase for consistency")
    #lowercase_all_words(conn)
    #print("Lowercased words")
#
    #print("Removing words with identical normalized_phoneme_list, keeping shortest word (rare cases)")
    #removed_norm_phonemes = remove_duplicates_by_normalized_phonemes(conn)
    #print(f"Removed words with identical normalized_phoneme_list: {removed_norm_phonemes}")
#

    print("Database cleaning complete.")
    conn.close()


if __name__ == '__main__':
    main()
