#!/usr/bin/env python3
"""
Script to trim the adapted_needham_dataset.json to only the first 20 samples.
This script reads the full dataset, keeps only the first 20 samples, and saves them
to a new file while also creating a backup of the original.
"""

import json
import os
from datetime import datetime

def trim_dataset():
    # Define paths
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'datasets',
        'adapted_needham_dataset.json'
    )
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'datasets',
        'tiny_adapted_needham_dataset.json'
    )
    
    backup_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'datasets',
        f'adapted_needham_dataset_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    print(f"Reading dataset from: {dataset_path}")
    
    # Read the original dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original dataset contains {len(data)} samples")
    
    # Keep only the first 20 samples
    trimmed_data = data[:40]
    
    print(f"Trimmed dataset contains {len(trimmed_data)} samples")
    
    # Save the trimmed dataset to a new file
    print(f"Saving trimmed dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trimmed_data, f, indent=2, ensure_ascii=False)
    
    print("\nDone! Summary:")
    print(f"  - Original samples: {len(data)}")
    print(f"  - Trimmed samples: {len(trimmed_data)}")
    print(f"  - Backup created: {backup_path}")
    print(f"  - New file created: {output_path}")
    print("\nNote: The original file has NOT been modified.")
    print("If you want to replace the original file with the trimmed version,")
    print(f"you can manually copy {output_path} to {dataset_path}")

if __name__ == "__main__":
    trim_dataset()

