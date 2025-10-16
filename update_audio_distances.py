#!/usr/bin/env python3
"""
One-time script to populate audio_based_distance column in phoneme_distances table
from phoneme_audio_distance.csv file.

This version ensures the table is symmetrical by inserting/updating both
(phoneme1, phoneme2) and (phoneme2, phoneme1) entries for each pair in the CSV.
This guarantees that lookups work regardless of phoneme order.
"""

import sqlite3
import csv
import os
from pathlib import Path

def update_audio_distances(db_path: str, csv_path: str):
    """
    Update the audio_based_distance column in phoneme_distances table
    using values from the CSV file.
    
    Args:
        db_path: Path to BestPhonetics.db
        csv_path: Path to phoneme_audio_distance.csv
    """
    
    # Verify files exist
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return False
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    print(f"Reading CSV file: {csv_path}")
    print(f"Updating database: {db_path}")
    
    # Read CSV data
    updates = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phoneme1 = row['Phoneme 1'].strip()
            phoneme2 = row['Phoneme 2'].strip()
            audio_dist = float(row['audio_distance'].strip())
            updates.append((audio_dist, phoneme1, phoneme2))
    
    print(f"Read {len(updates)} rows from CSV")
    
    # Connect to database and update
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First, check if the audio_based_distance column exists
    cursor.execute("PRAGMA table_info(phoneme_distances)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'audio_based_distance' not in columns:
        print("Error: audio_based_distance column does not exist in phoneme_distances table")
        print(f"Available columns: {', '.join(columns)}")
        conn.close()
        return False
    
    print("Column exists, proceeding with updates...")
    print()
    
    # For each phoneme pair, we need to ensure BOTH orderings exist in the table
    # SQL to insert or update a phoneme pair
    upsert_sql = """
        INSERT INTO phoneme_distances (phoneme_1, phoneme_2, feature_based_distance, audio_based_distance)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(phoneme_1, phoneme_2) 
        DO UPDATE SET audio_based_distance = excluded.audio_based_distance
    """
    
    # First pass: get feature_based_distance for each pair if it exists
    feature_distances = {}
    for audio_dist, phoneme1, phoneme2 in updates:
        # Check both orderings to see if a feature distance exists
        cursor.execute("""
            SELECT feature_based_distance 
            FROM phoneme_distances 
            WHERE (phoneme_1 = ? AND phoneme_2 = ?) OR (phoneme_1 = ? AND phoneme_2 = ?)
        """, (phoneme1, phoneme2, phoneme2, phoneme1))
        
        result = cursor.fetchone()
        if result and result[0] is not None:
            feature_distances[(phoneme1, phoneme2)] = result[0]
            feature_distances[(phoneme2, phoneme1)] = result[0]
        else:
            # Default to 0.0 if no feature distance exists
            feature_distances[(phoneme1, phoneme2)] = 0.0
            feature_distances[(phoneme2, phoneme1)] = 0.0
    
    updated_count = 0
    inserted_count = 0
    
    for audio_dist, phoneme1, phoneme2 in updates:
        # Insert/update both orderings to ensure symmetry
        pairs_to_update = [
            (phoneme1, phoneme2),
            (phoneme2, phoneme1)
        ]
        
        for p1, p2 in pairs_to_update:
            feature_dist = feature_distances.get((p1, p2), 0.0)
            
            # Check if row exists
            cursor.execute("""
                SELECT COUNT(*) FROM phoneme_distances 
                WHERE phoneme_1 = ? AND phoneme_2 = ?
            """, (p1, p2))
            
            exists = cursor.fetchone()[0] > 0
            
            # Insert or update
            cursor.execute(upsert_sql, (p1, p2, feature_dist, audio_dist))
            
            if exists:
                updated_count += 1
            else:
                inserted_count += 1
    
    conn.commit()
    
    print()
    print("Results:")
    print(f"  Rows updated: {updated_count}")
    print(f"  Rows inserted: {inserted_count}")
    print(f"  Total operations: {updated_count + inserted_count}")
    print()
    print(f"  CSV entries processed: {len(updates)}")
    print(f"  Database entries created/updated: {(updated_count + inserted_count) // 2} pairs (symmetrical)")
    
    # Verify some updates
    cursor.execute("""
        SELECT phoneme_1, phoneme_2, audio_based_distance 
        FROM phoneme_distances 
        WHERE audio_based_distance IS NOT NULL 
        ORDER BY phoneme_1, phoneme_2
        LIMIT 5
    """)
    
    print()
    print("Sample of updated rows:")
    for row in cursor.fetchall():
        print(f"  {row[0]} - {row[1]}: {row[2]}")
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM phoneme_distances WHERE audio_based_distance IS NOT NULL")
    total_updated = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM phoneme_distances")
    total_rows = cursor.fetchone()[0]
    
    print()
    print("Database statistics:")
    print(f"  Total rows in table: {total_rows}")
    print(f"  Rows with audio_based_distance: {total_updated}")
    print(f"  Rows without audio_based_distance: {total_rows - total_updated}")
    
    conn.close()
    
    print()
    print("✓ Update complete!")
    return True


if __name__ == "__main__":
    # Use paths relative to script location
    script_dir = Path(__file__).parent
    
    db_path = script_dir / "data" / "BestPhonetics.db"
    csv_path = script_dir / "phoneme_audio_distance.csv"
    
    print("=" * 70)
    print("Audio Distance Update Script (v3 - symmetrical table)")
    print("=" * 70)
    print()
    
    success = update_audio_distances(str(db_path), str(csv_path))
    
    if not success:
        print()
        print("❌ Update failed!")
        exit(1)
    else:
        print()
        print("✓ All done!")
        exit(0)
