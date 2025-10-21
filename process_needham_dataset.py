"""
Process the Needham dataset to extract and format conversation samples.

This script processes raw conversation data from the Needham dataset and transforms
it into a standardized format suitable for evaluation. It:
- Formats messages for OpenAI API compatibility
- Computes eval_human scores from multiple author ratings
- Filters and cleans metadata fields
"""

import json
from typing import Any, Dict, List


def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single message to format it for OpenAI API.
    Removes the 'id' field and keeps only relevant fields.
    """
    processed = {
        "role": message["role"],
        "content": message["content"]
    }
    
    # Keep tool_calls if present (for assistant messages with tool calls)
    if "tool_calls" in message:
        processed["tool_calls"] = message["tool_calls"]
    
    # Keep tool_call_id if present (for tool response messages)
    if "tool_call_id" in message:
        processed["tool_call_id"] = message["tool_call_id"]
    
    return processed


def process_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process metadata to keep only specified fields and compute eval_human.
    """
    eval_author1 = metadata.get("eval_author1", 0)
    eval_author2 = metadata.get("eval_author2", 0)
    eval_human = round((eval_author1 + eval_author2) / 2, 4)
    
    processed_metadata = {
        "category": metadata.get("category"),
        "eval_category": metadata.get("eval_category"),
        "safety_category": metadata.get("safety_category"),
        "tools": metadata.get("tools"),
        "eval_type": metadata.get("eval_type"),
        "eval_human": eval_human
    }
    
    return processed_metadata


def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single sample from the dataset.
    """
    processed_sample = {
        "id": sample["id"],
        "input": [process_message(msg) for msg in sample["input"]],
        "metadata": process_metadata(sample["metadata"])
    }
    
    return processed_sample


def main():
    input_file = "/Users/igorivanov/Cursor_projects/Apollo scheming evals/Igor's project/datasets/needham_dataset.json"
    output_file = "/Users/igorivanov/Cursor_projects/Apollo scheming evals/Igor's project/datasets/needham_dataset_processed.json"
    
    print("Reading dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Processing {len(dataset)} samples...")
    processed_dataset = [process_sample(sample) for sample in dataset]
    
    print("Writing processed dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Processed dataset saved to: {output_file}")
    print(f"Total samples processed: {len(processed_dataset)}")


if __name__ == "__main__":
    main()

