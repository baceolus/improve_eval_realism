"""
Order Consistency Experiment

This experiment tests whether the compare_transcripts_realism function shows
positional bias within each of its 5 individual judge prompts.

Since the new compare_transcripts_realism already swaps transcript order internally
for each prompt, we analyze the individual results from each comparison to check:
1. Whether each of the 5 prompts shows position bias (preferring first or second position)
2. Whether all 5 prompts show consistent bias in the same direction

Positional bias is detected when a prompt consistently favors one position over another
when comparing the same pair of transcripts in both orders.
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import from other modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from calculate_leaderboard import load_dataset, AGENTIC_GROUPS
from sample_comparison import extract_conversation_to_string, compare_transcripts_realism


def analyze_prompt_bias(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze position bias for each of the 5 prompts from compare_transcripts_realism results.
    
    For each prompt, we have 2 results: original order and swapped order.
    Position bias occurs when the prompt consistently favors one position.
    
    Args:
        individual_results: List of 10 individual comparison results from compare_transcripts_realism
        
    Returns:
        Dictionary with bias analysis for each prompt
    """
    # Group results by prompt_index
    prompt_results = {}
    for result_item in individual_results:
        prompt_idx = result_item["prompt_index"]
        if prompt_idx not in prompt_results:
            prompt_results[prompt_idx] = {"original": None, "swapped": None}
        
        order = result_item["order"]
        prompt_results[prompt_idx][order] = result_item["result"].get("more_realistic", "error")
    
    # Analyze bias for each prompt
    prompt_bias_analysis = {}
    
    for prompt_idx, results in prompt_results.items():
        original_result = results["original"]
        swapped_result = results["swapped"]
        
        # Check for position bias
        # The swapped result is already adjusted in compare_transcripts_realism
        # so "1" always means transcript1 wins, "2" means transcript2 wins
        # 
        # Original order: transcript1 in position 1, transcript2 in position 2
        #   - Result "1" means first position wins
        #   - Result "2" means second position wins
        # 
        # Swapped order (adjusted): transcript2 in position 1, transcript1 in position 2
        #   - Result "1" means transcript1 wins (which is in second position)
        #   - Result "2" means transcript2 wins (which is in first position)
        
        # Count position preferences
        first_position_preference = 0  # How many times first position was chosen
        second_position_preference = 0  # How many times second position was chosen
        tie_count = 0
        
        # Original order
        if original_result == "1":
            first_position_preference += 1
        elif original_result == "2":
            second_position_preference += 1
        elif original_result == "tie":
            tie_count += 1
        
        # Swapped order (already adjusted, so "1" means transcript1, "2" means transcript2)
        if swapped_result == "1":
            # transcript1 won, but it's in second position now
            second_position_preference += 1
        elif swapped_result == "2":
            # transcript2 won, but it's in first position now
            first_position_preference += 1
        elif swapped_result == "tie":
            tie_count += 1
        
        # Determine bias type
        if first_position_preference > second_position_preference:
            bias_type = "first_position"
            bias_strength = first_position_preference - second_position_preference
        elif second_position_preference > first_position_preference:
            bias_type = "second_position"
            bias_strength = second_position_preference - first_position_preference
        else:
            bias_type = "none"
            bias_strength = 0
        
        prompt_bias_analysis[prompt_idx] = {
            "first_position_count": first_position_preference,
            "second_position_count": second_position_preference,
            "tie_count": tie_count,
            "bias_type": bias_type,
            "bias_strength": bias_strength,
            "original_result": original_result,
            "swapped_result": swapped_result
        }
    
    return prompt_bias_analysis


def check_consistent_bias_across_prompts(prompt_bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze how many prompts show no bias vs some bias.
    
    Args:
        prompt_bias_analysis: Bias analysis for each prompt
        
    Returns:
        Dictionary with bias distribution analysis
    """
    bias_types = [analysis["bias_type"] for analysis in prompt_bias_analysis.values()]
    
    # Count bias types
    first_position_count = sum(1 for bt in bias_types if bt == "first_position")
    second_position_count = sum(1 for bt in bias_types if bt == "second_position")
    no_bias_count = sum(1 for bt in bias_types if bt == "none")
    
    # For backward compatibility, check if all prompts show same bias
    has_consistent_bias = False
    consistent_bias_direction = "none"
    
    if first_position_count == 5:
        has_consistent_bias = True
        consistent_bias_direction = "first_position"
    elif second_position_count == 5:
        has_consistent_bias = True
        consistent_bias_direction = "second_position"
    
    return {
        "has_consistent_bias": has_consistent_bias,
        "consistent_bias_direction": consistent_bias_direction,
        "first_position_bias_count": first_position_count,
        "second_position_bias_count": second_position_count,
        "no_bias_count": no_bias_count,
        "unbiased_prompts_count": no_bias_count,  # Number of prompts with no bias
        "all_bias_types": bias_types
    }


def process_pair(
    pair_number: int,
    sample1: Dict[str, Any],
    sample2: Dict[str, Any],
    api_key: str,
    model: str
) -> Dict[str, Any]:
    """
    Process a single pair using the new compare_transcripts_realism function.
    Analyzes position bias within each of the 5 prompts.
    
    Args:
        pair_number: The pair number (for tracking)
        sample1: First sample
        sample2: Second sample
        api_key: OpenRouter API key
        model: Model to use for comparisons
        
    Returns:
        Dictionary with pair results including bias analysis
    """
    sample1_id = sample1.get('id')
    sample2_id = sample2.get('id')
    
    try:
        # Extract transcripts
        transcript1 = extract_conversation_to_string(sample1)
        transcript2 = extract_conversation_to_string(sample2)
        
        # Run comparison (internally does 10 comparisons: 5 prompts × 2 orders)
        comparison_result = compare_transcripts_realism(transcript1, transcript2, api_key, model)
        
        # Analyze position bias for each prompt
        prompt_bias_analysis = analyze_prompt_bias(comparison_result["individual_results"])
        
        # Check for consistent bias across all prompts
        consistent_bias_check = check_consistent_bias_across_prompts(prompt_bias_analysis)
        
        # Record result
        pair_result = {
            "pair_number": pair_number,
            "sample1_id": sample1_id,
            "sample2_id": sample2_id,
            "final_winner": comparison_result["final_winner"],
            "vote_counts": comparison_result["vote_counts"],
            "prompt_bias_analysis": prompt_bias_analysis,
            "consistent_bias_check": consistent_bias_check,
            "full_comparison_result": comparison_result
        }
        
        return pair_result
        
    except Exception as e:
        pair_result = {
            "pair_number": pair_number,
            "sample1_id": sample1_id,
            "sample2_id": sample2_id,
            "error": str(e)
        }
        return pair_result


def run_order_consistency_experiment(
    samples: List[Dict[str, Any]],
    api_key: str,
    model: str,
    n_pairs: int = 100,
    random_seed: int = 42,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Run position bias analysis experiment on random sample pairs in parallel.
    
    For each pair, analyzes whether each of the 5 judge prompts shows position bias
    and whether all 5 prompts consistently show bias in the same direction.
    
    Args:
        samples: List of samples to test
        api_key: OpenRouter API key
        model: Model to use for comparisons
        n_pairs: Number of pairs to test
        random_seed: Random seed for reproducibility
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with experiment results including:
        - per_prompt_bias: aggregated bias statistics for each prompt
        - consistent_bias_across_prompts: how often all 5 prompts show the same bias
        - pair_results: detailed results for each pair
    """
    print(f"\n{'='*70}")
    print(f"Position Bias Analysis Experiment")
    print(f"{'='*70}")
    print(f"Samples available: {len(samples)}")
    print(f"Number of pairs to test: {n_pairs}")
    print(f"Random seed: {random_seed}")
    print(f"Model: {model}")
    print(f"Max parallel workers: {max_workers}")
    print(f"{'='*70}\n")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Pre-generate all sample pairs
    print(f"Generating {n_pairs} random sample pairs...")
    sample_pairs = []
    for i in range(n_pairs):
        sample1, sample2 = random.sample(samples, 2)
        sample_pairs.append((i + 1, sample1, sample2))
    
    print(f"Running {n_pairs} pair comparisons in parallel...\n")
    
    # Process all pairs in parallel
    pair_results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pair = {
            executor.submit(
                process_pair,
                pair_number, sample1, sample2, api_key, model
            ): (pair_number, sample1.get('id'), sample2.get('id'))
            for pair_number, sample1, sample2 in sample_pairs
        }
        
        # Process results as they complete
        for future in as_completed(future_to_pair):
            pair_number, sample1_id, sample2_id = future_to_pair[future]
            pair_result = future.result()
            pair_results.append(pair_result)
            
            completed_count += 1
            
            # Print progress
            if 'error' in pair_result:
                print(f"  [{completed_count}/{n_pairs}] Pair {pair_number}: ✗ ERROR - {pair_result['error'][:50]}...")
            else:
                winner = pair_result['final_winner']
                has_consistent = pair_result['consistent_bias_check']['has_consistent_bias']
                bias_dir = pair_result['consistent_bias_check']['consistent_bias_direction']
                status = f"Winner: {winner}"
                if has_consistent:
                    status += f" | ⚠ ALL PROMPTS BIASED: {bias_dir}"
                print(f"  [{completed_count}/{n_pairs}] Pair {pair_number} ({sample1_id} vs {sample2_id}): {status}")
    
    # Sort results by pair number
    pair_results.sort(key=lambda x: x['pair_number'])
    
    # Aggregate statistics
    valid_pairs = [pr for pr in pair_results if 'error' not in pr]
    
    # Per-prompt bias statistics
    per_prompt_bias = {i: {"first_position": 0, "second_position": 0, "none": 0} for i in range(1, 6)}
    
    for pair_result in valid_pairs:
        for prompt_idx, bias_info in pair_result['prompt_bias_analysis'].items():
            bias_type = bias_info['bias_type']
            per_prompt_bias[prompt_idx][bias_type] += 1
    
    # Distribution of unbiased prompts across pairs
    # Count how many pairs have 5, 4, 3, 2, 1, or 0 unbiased prompts
    unbiased_prompts_distribution = {
        5: 0,  # All 5 prompts unbiased
        4: 0,  # 4 prompts unbiased, 1 biased
        3: 0,  # 3 prompts unbiased, 2 biased
        2: 0,  # 2 prompts unbiased, 3 biased
        1: 0,  # 1 prompt unbiased, 4 biased
        0: 0   # All 5 prompts biased
    }
    
    # Legacy stats for backward compatibility
    consistent_bias_stats = {
        "all_first_position": 0,
        "all_second_position": 0,
        "mixed_or_no_bias": 0
    }
    
    for pair_result in valid_pairs:
        consistent_check = pair_result['consistent_bias_check']
        
        # Count unbiased prompts for this pair
        unbiased_count = consistent_check.get('unbiased_prompts_count', 0)
        if unbiased_count in unbiased_prompts_distribution:
            unbiased_prompts_distribution[unbiased_count] += 1
        
        # Legacy stats
        if consistent_check['has_consistent_bias']:
            if consistent_check['consistent_bias_direction'] == "first_position":
                consistent_bias_stats["all_first_position"] += 1
            elif consistent_check['consistent_bias_direction'] == "second_position":
                consistent_bias_stats["all_second_position"] += 1
        else:
            consistent_bias_stats["mixed_or_no_bias"] += 1
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Experiment Results")
    print(f"{'='*70}")
    print(f"Total pairs tested: {n_pairs}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"\nPer-Prompt Bias Analysis:")
    print(f"-" * 70)
    
    for prompt_idx in range(1, 6):
        stats = per_prompt_bias[prompt_idx]
        total = sum(stats.values())
        if total > 0:
            print(f"  Prompt {prompt_idx}:")
            print(f"    First position bias:  {stats['first_position']:3d} ({stats['first_position']/total*100:5.1f}%)")
            print(f"    Second position bias: {stats['second_position']:3d} ({stats['second_position']/total*100:5.1f}%)")
            print(f"    No bias:              {stats['none']:3d} ({stats['none']/total*100:5.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"Distribution of Unbiased Prompts:")
    print(f"-" * 70)
    total_valid = len(valid_pairs)
    if total_valid > 0:
        print(f"  5 prompts unbiased (0 biased): {unbiased_prompts_distribution[5]:3d} ({unbiased_prompts_distribution[5]/total_valid*100:5.1f}%)")
        print(f"  4 prompts unbiased (1 biased): {unbiased_prompts_distribution[4]:3d} ({unbiased_prompts_distribution[4]/total_valid*100:5.1f}%)")
        print(f"  3 prompts unbiased (2 biased): {unbiased_prompts_distribution[3]:3d} ({unbiased_prompts_distribution[3]/total_valid*100:5.1f}%)")
        print(f"  2 prompts unbiased (3 biased): {unbiased_prompts_distribution[2]:3d} ({unbiased_prompts_distribution[2]/total_valid*100:5.1f}%)")
        print(f"  1 prompt unbiased  (4 biased): {unbiased_prompts_distribution[1]:3d} ({unbiased_prompts_distribution[1]/total_valid*100:5.1f}%)")
        print(f"  0 prompts unbiased (5 biased): {unbiased_prompts_distribution[0]:3d} ({unbiased_prompts_distribution[0]/total_valid*100:5.1f}%)")
        
        if unbiased_prompts_distribution[0] > 0:
            print(f"\n  ⚠ WARNING: {unbiased_prompts_distribution[0]} pair(s) have all 5 prompts showing bias!")
    
    print(f"{'='*70}\n")
    
    # Prepare final results
    results = {
        "experiment_info": {
            "type": "position_bias_analysis",
            "model": model,
            "n_pairs": n_pairs,
            "random_seed": random_seed,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "total_pairs": n_pairs,
            "valid_pairs": len(valid_pairs),
            "per_prompt_bias": per_prompt_bias,
            "unbiased_prompts_distribution": unbiased_prompts_distribution,
            "consistent_bias_stats": consistent_bias_stats  # Legacy
        },
        "pair_results": pair_results
    }
    
    return results


def main():
    # Load environment from igor_project/.env
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    
    # Configuration
    model = "openai/gpt-5-mini"
    n_pairs = 100
    random_seed = 42
    max_workers = 100  # Number of parallel workers
    
    # Load dataset
    print("Loading dataset...")
    samples = load_dataset()
    
    # Filter for agentic samples only
    agentic_samples = [s for s in samples if s.get('metadata', {}).get('category') in AGENTIC_GROUPS]
    
    print(f"Total samples: {len(samples)}")
    print(f"Agentic samples: {len(agentic_samples)}")
    
    if len(agentic_samples) < 2:
        print("ERROR: Not enough agentic samples to run experiment")
        sys.exit(1)
    
    # Run experiment
    results = run_order_consistency_experiment(
        samples=agentic_samples,
        api_key=api_key,
        model=model,
        n_pairs=n_pairs,
        random_seed=random_seed,
        max_workers=max_workers
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"position_bias_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    # Also save a summary report
    summary_file = output_dir / f"position_bias_summary_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Position Bias Analysis Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {results['experiment_info']['timestamp']}\n")
        f.write(f"Model: {results['experiment_info']['model']}\n")
        f.write(f"Random seed: {results['experiment_info']['random_seed']}\n\n")
        f.write(f"Total pairs tested: {results['summary']['total_pairs']}\n")
        f.write(f"Valid pairs: {results['summary']['valid_pairs']}\n\n")
        
        # Per-prompt bias statistics
        f.write("-"*70 + "\n")
        f.write("Per-Prompt Bias Analysis:\n")
        f.write("-"*70 + "\n")
        
        per_prompt_bias = results['summary']['per_prompt_bias']
        for prompt_idx in range(1, 6):
            stats = per_prompt_bias[prompt_idx]
            total = sum(stats.values())
            if total > 0:
                f.write(f"\nPrompt {prompt_idx}:\n")
                f.write(f"  First position bias:  {stats['first_position']:3d} ({stats['first_position']/total*100:5.1f}%)\n")
                f.write(f"  Second position bias: {stats['second_position']:3d} ({stats['second_position']/total*100:5.1f}%)\n")
                f.write(f"  No bias:              {stats['none']:3d} ({stats['none']/total*100:5.1f}%)\n")
        
        # Distribution of unbiased prompts
        f.write("\n" + "="*70 + "\n")
        f.write("Distribution of Unbiased Prompts:\n")
        f.write("="*70 + "\n")
        
        unbiased_dist = results['summary']['unbiased_prompts_distribution']
        total_valid = results['summary']['valid_pairs']
        if total_valid > 0:
            f.write(f"5 prompts unbiased (0 biased): {unbiased_dist[5]:3d} ({unbiased_dist[5]/total_valid*100:5.1f}%)\n")
            f.write(f"4 prompts unbiased (1 biased): {unbiased_dist[4]:3d} ({unbiased_dist[4]/total_valid*100:5.1f}%)\n")
            f.write(f"3 prompts unbiased (2 biased): {unbiased_dist[3]:3d} ({unbiased_dist[3]/total_valid*100:5.1f}%)\n")
            f.write(f"2 prompts unbiased (3 biased): {unbiased_dist[2]:3d} ({unbiased_dist[2]/total_valid*100:5.1f}%)\n")
            f.write(f"1 prompt unbiased  (4 biased): {unbiased_dist[1]:3d} ({unbiased_dist[1]/total_valid*100:5.1f}%)\n")
            f.write(f"0 prompts unbiased (5 biased): {unbiased_dist[0]:3d} ({unbiased_dist[0]/total_valid*100:5.1f}%)\n")
            
            if unbiased_dist[0] > 0:
                f.write(f"\n⚠ WARNING: {unbiased_dist[0]} pair(s) have all 5 prompts showing bias!\n")
        
        # Detailed results
        f.write("\n" + "="*70 + "\n")
        f.write("Detailed Results\n")
        f.write("="*70 + "\n\n")
        
        for pair in results['pair_results']:
            if 'error' in pair:
                f.write(f"Pair {pair['pair_number']}: ERROR - {pair['error']}\n")
            else:
                f.write(f"Pair {pair['pair_number']}:\n")
                f.write(f"  Samples: {pair['sample1_id']} vs {pair['sample2_id']}\n")
                f.write(f"  Final winner: {pair['final_winner']}\n")
                f.write(f"  Vote counts: {pair['vote_counts']}\n")
                
                # Show per-prompt bias
                f.write(f"  Per-prompt bias:\n")
                for prompt_idx, bias_info in pair['prompt_bias_analysis'].items():
                    f.write(f"    Prompt {prompt_idx}: {bias_info['bias_type']} ")
                    f.write(f"(orig: {bias_info['original_result']}, swap: {bias_info['swapped_result']})\n")
                
                # Show if consistent bias
                consistent_check = pair['consistent_bias_check']
                if consistent_check['has_consistent_bias']:
                    f.write(f"  ⚠ CONSISTENT BIAS: All prompts favor {consistent_check['consistent_bias_direction']}\n")
                f.write("\n")
    
    print(f"✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

