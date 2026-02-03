"""
Helper script to copy dataset from Downloads folder to data/ folder.
"""

import shutil
from pathlib import Path
import os

def find_and_copy_dataset():
    """Find dataset in Downloads and copy to data folder."""
    downloads_path = Path.home() / "Downloads"
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    
    # Look for cybersecurity dataset files
    patterns = [
        "*cybersecurity*attack*.csv",
        "*cybersecurity*attack*.xlsx",
        "*cybersecurity*attack*.xls",
        "*attack*.csv",
        "*attack*.xlsx"
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(list(downloads_path.glob(pattern)))
    
    if not found_files:
        print("[ERROR] No dataset files found in Downloads folder.")
        print("\nPlease manually copy your dataset file to the 'data/' folder.")
        print("Supported formats: CSV, Excel (.xlsx, .xls)")
        return False
    
    # Use the first found file
    source_file = found_files[0]
    dest_file = data_folder / source_file.name
    
    try:
        shutil.copy2(source_file, dest_file)
        print(f"[OK] Dataset copied successfully!")
        print(f"   Source: {source_file}")
        print(f"   Destination: {dest_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Error copying file: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Dataset Setup Helper")
    print("="*60)
    find_and_copy_dataset()

