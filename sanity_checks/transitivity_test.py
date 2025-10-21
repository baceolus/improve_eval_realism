"""
Transitivity Test for compare_transcripts_realism

This experiment tests whether the compare_transcripts_realism function exhibits
transitivity in its comparisons. Transitivity means that if sample A is judged
more realistic than sample B, and sample B is more realistic than sample C,
then sample A should be more realistic than sample C.

The test:
1. Randomly selects triplets of samples
2. Runs compare_transcripts_realism on all 3 pairs: (A,B), (B,C), (A,C)
3. Checks if transitivity holds (accounting for ties)
4. Calculates the percentage of triplets that pass the transitivity test
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


def check_transitivity(result_ab: str, result_bc: str, result_ac: str) -> Tuple[bool, str]:
    """
    Check if the comparison results satisfy transitivity.
    
    Transitivity with ties means:
    - If A ≥ B (A wins or ties) and B ≥ C (B wins or ties), then A ≥ C (A wins or ties)
    
    Args:
        result_ab: Result of comparing A vs B ("1" means A wins, "2" means B wins, "tie")
        result_bc: Result of comparing B vs C ("1" means B wins, "2" means C wins, "tie")
        result_ac: Result of comparing A vs C ("1" means A wins, "2" means C wins, "tie")
        
    Returns:
        Tuple of (passes_transitivity: bool, violation_description: str)
    """
    # Convert results to more readable form
    # For AB: "1" means A>B, "2" means B>A, "tie" means A=B
    # For BC: "1" means B>C, "2" means C>B, "tie" means B=C
    # For AC: "1" means A>C, "2" means C>A, "tie" means A=C
    
    # Map results
    ab = "A>B" if result_ab == "1" else ("B>A" if result_ab == "2" else "A=B")
    bc = "B>C" if result_bc == "1" else ("C>B" if result_bc == "2" else "B=C")
    ac = "A>C" if result_ac == "1" else ("C>A" if result_ac == "2" else "A=C")
    
    # Check all transitivity rules
    # Rule: If A ≥ B and B ≥ C, then A ≥ C
    
    # Case 1: A > B and B > C, then A must be ≥ C (i.e., not C > A)
    if result_ab == "1" and result_bc == "1":
        if result_ac == "2":  # C > A violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 2: A > B and B = C, then A must be ≥ C (i.e., not C > A)
    if result_ab == "1" and result_bc == "tie":
        if result_ac == "2":  # C > A violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 3: A = B and B > C, then A must be ≥ C (i.e., not C > A)
    if result_ab == "tie" and result_bc == "1":
        if result_ac == "2":  # C > A violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 4: A = B and B = C, then A must = C
    if result_ab == "tie" and result_bc == "tie":
        if result_ac != "tie":  # A≠C violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 5: B > A and C > B, then C must be ≥ A (i.e., not A > C)
    if result_ab == "2" and result_bc == "2":
        if result_ac == "1":  # A > C violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 6: B > A and B = C, then C must be ≥ A (i.e., not A > C)
    if result_ab == "2" and result_bc == "tie":
        if result_ac == "1":  # A > C violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 7: A = B and C > B, then C must be ≥ A (i.e., not A > C)
    if result_ab == "tie" and result_bc == "2":
        if result_ac == "1":  # A > C violates transitivity
            return False, f"Violation: {ab} and {bc}, but {ac}"
        return True, f"Valid: {ab} and {bc}, and {ac}"
    
    # Case 8: B > A and B > C - no constraint on A vs C
    if result_ab == "2" and result_bc == "1":
        return True, f"No constraint: {ab} and {bc}, {ac} is acceptable"
    
    # Case 9: A > B and C > B - no constraint on A vs C
    if result_ab == "1" and result_bc == "2":
        return True, f"No constraint: {ab} and {bc}, {ac} is acceptable"
    
    # Shouldn't reach here, but just in case
    return True, f"Uncategorized: {ab}, {bc}, {ac}"


def process_triplet(
    triplet_number: int,
    sample_a: Dict[str, Any],
    sample_b: Dict[str, Any],
    sample_c: Dict[str, Any],
    api_key: str,
    model: str
) -> Dict[str, Any]:
    """
    Process a single triplet of samples by comparing all three pairs.
    
    Args:
        triplet_number: The triplet number (for tracking)
        sample_a: First sample (A)
        sample_b: Second sample (B)
        sample_c: Third sample (C)
        api_key: OpenRouter API key
        model: Model to use for comparisons
        
    Returns:
        Dictionary with triplet results including transitivity check
    """
    sample_a_id = sample_a.get('id')
    sample_b_id = sample_b.get('id')
    sample_c_id = sample_c.get('id')
    
    try:
        # Extract transcripts
        transcript_a = extract_conversation_to_string(sample_a)
        transcript_b = extract_conversation_to_string(sample_b)
        transcript_c = extract_conversation_to_string(sample_c)
        
        # Run all three pairwise comparisons
        # A vs B
        comparison_ab = compare_transcripts_realism(transcript_a, transcript_b, api_key, model)
        result_ab = comparison_ab["final_winner"]
        
        # B vs C
        comparison_bc = compare_transcripts_realism(transcript_b, transcript_c, api_key, model)
        result_bc = comparison_bc["final_winner"]
        
        # A vs C
        comparison_ac = compare_transcripts_realism(transcript_a, transcript_c, api_key, model)
        result_ac = comparison_ac["final_winner"]
        
        # Check transitivity
        passes_transitivity, explanation = check_transitivity(result_ab, result_bc, result_ac)
        
        # Record result
        triplet_result = {
            "triplet_number": triplet_number,
            "sample_a_id": sample_a_id,
            "sample_b_id": sample_b_id,
            "sample_c_id": sample_c_id,
            "result_ab": result_ab,
            "result_bc": result_bc,
            "result_ac": result_ac,
            "passes_transitivity": passes_transitivity,
            "explanation": explanation,
            "comparison_ab": comparison_ab,
            "comparison_bc": comparison_bc,
            "comparison_ac": comparison_ac
        }
        
        return triplet_result
        
    except Exception as e:
        triplet_result = {
            "triplet_number": triplet_number,
            "sample_a_id": sample_a_id,
            "sample_b_id": sample_b_id,
            "sample_c_id": sample_c_id,
            "error": str(e)
        }
        return triplet_result


def run_transitivity_experiment(
    samples: List[Dict[str, Any]],
    api_key: str,
    model: str,
    n_triplets: int = 100,
    random_seed: int = 42,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Run transitivity test experiment on random sample triplets in parallel.
    
    For each triplet (A, B, C), compares all three pairs and checks if
    transitivity holds.
    
    Args:
        samples: List of samples to test
        api_key: OpenRouter API key
        model: Model to use for comparisons
        n_triplets: Number of triplets to test
        random_seed: Random seed for reproducibility
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with experiment results including transitivity pass rate
    """
    print(f"\n{'='*70}")
    print(f"Transitivity Test Experiment")
    print(f"{'='*70}")
    print(f"Samples available: {len(samples)}")
    print(f"Number of triplets to test: {n_triplets}")
    print(f"Random seed: {random_seed}")
    print(f"Model: {model}")
    print(f"Max parallel workers: {max_workers}")
    print(f"{'='*70}\n")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Pre-generate all sample triplets
    print(f"Generating {n_triplets} random sample triplets...")
    sample_triplets = []
    for i in range(n_triplets):
        sample_a, sample_b, sample_c = random.sample(samples, 3)
        sample_triplets.append((i + 1, sample_a, sample_b, sample_c))
    
    print(f"Running {n_triplets} triplet comparisons in parallel...")
    print(f"(Each triplet requires 3 comparisons, so {n_triplets * 3} total comparisons)\n")
    
    # Process all triplets in parallel
    triplet_results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_triplet = {
            executor.submit(
                process_triplet,
                triplet_number, sample_a, sample_b, sample_c, api_key, model
            ): (triplet_number, sample_a.get('id'), sample_b.get('id'), sample_c.get('id'))
            for triplet_number, sample_a, sample_b, sample_c in sample_triplets
        }
        
        # Process results as they complete
        for future in as_completed(future_to_triplet):
            triplet_number, sample_a_id, sample_b_id, sample_c_id = future_to_triplet[future]
            triplet_result = future.result()
            triplet_results.append(triplet_result)
            
            completed_count += 1
            
            # Print progress
            if 'error' in triplet_result:
                print(f"  [{completed_count}/{n_triplets}] Triplet {triplet_number}: ✗ ERROR - {triplet_result['error'][:50]}...")
            else:
                passes = "✓ PASS" if triplet_result['passes_transitivity'] else "✗ FAIL"
                ab = triplet_result['result_ab']
                bc = triplet_result['result_bc']
                ac = triplet_result['result_ac']
                print(f"  [{completed_count}/{n_triplets}] Triplet {triplet_number} ({sample_a_id}, {sample_b_id}, {sample_c_id}): {passes} | AB:{ab} BC:{bc} AC:{ac}")
    
    # Sort results by triplet number
    triplet_results.sort(key=lambda x: x['triplet_number'])
    
    # Calculate statistics
    valid_triplets = [tr for tr in triplet_results if 'error' not in tr]
    passed_triplets = [tr for tr in valid_triplets if tr['passes_transitivity']]
    failed_triplets = [tr for tr in valid_triplets if not tr['passes_transitivity']]
    
    pass_rate = len(passed_triplets) / len(valid_triplets) if valid_triplets else 0.0
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Experiment Results")
    print(f"{'='*70}")
    print(f"Total triplets tested: {n_triplets}")
    print(f"Valid triplets: {len(valid_triplets)}")
    print(f"Triplets with errors: {n_triplets - len(valid_triplets)}")
    print(f"\nTransitivity Results:")
    print(f"  Passed: {len(passed_triplets)} ({pass_rate*100:.1f}%)")
    print(f"  Failed: {len(failed_triplets)} ({(1-pass_rate)*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if failed_triplets:
        print(f"Failed Triplet Examples:")
        print(f"-" * 70)
        for i, triplet in enumerate(failed_triplets[:5], 1):  # Show first 5 failures
            print(f"\nFailure {i}: Triplet {triplet['triplet_number']}")
            print(f"  Samples: {triplet['sample_a_id']}, {triplet['sample_b_id']}, {triplet['sample_c_id']}")
            print(f"  Results: AB={triplet['result_ab']}, BC={triplet['result_bc']}, AC={triplet['result_ac']}")
            print(f"  {triplet['explanation']}")
        if len(failed_triplets) > 5:
            print(f"\n  ... and {len(failed_triplets) - 5} more failures")
        print(f"{'='*70}\n")
    
    # Prepare final results
    results = {
        "experiment_info": {
            "type": "transitivity_test",
            "model": model,
            "n_triplets": n_triplets,
            "random_seed": random_seed,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "total_triplets": n_triplets,
            "valid_triplets": len(valid_triplets),
            "error_triplets": n_triplets - len(valid_triplets),
            "passed_triplets": len(passed_triplets),
            "failed_triplets": len(failed_triplets),
            "pass_rate": pass_rate
        },
        "triplet_results": triplet_results
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
    n_triplets = 100
    random_seed = 42
    max_workers = 300  # Number of parallel workers
    
    # Load dataset
    print("Loading dataset...")
    samples = load_dataset()
    
    # Filter for agentic samples only
    agentic_samples = [s for s in samples if s.get('metadata', {}).get('category') in AGENTIC_GROUPS]
    
    print(f"Total samples: {len(samples)}")
    print(f"Agentic samples: {len(agentic_samples)}")
    
    if len(agentic_samples) < 3:
        print("ERROR: Not enough agentic samples to run experiment (need at least 3)")
        sys.exit(1)
    
    # Run experiment
    results = run_transitivity_experiment(
        samples=agentic_samples,
        api_key=api_key,
        model=model,
        n_triplets=n_triplets,
        random_seed=random_seed,
        max_workers=max_workers
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"transitivity_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    # Also save a summary report
    summary_file = output_dir / f"transitivity_summary_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Transitivity Test Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {results['experiment_info']['timestamp']}\n")
        f.write(f"Model: {results['experiment_info']['model']}\n")
        f.write(f"Random seed: {results['experiment_info']['random_seed']}\n\n")
        f.write(f"Total triplets tested: {results['summary']['total_triplets']}\n")
        f.write(f"Valid triplets: {results['summary']['valid_triplets']}\n")
        f.write(f"Error triplets: {results['summary']['error_triplets']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Transitivity Results:\n")
        f.write("-"*70 + "\n")
        pass_rate = results['summary']['pass_rate']
        f.write(f"Passed: {results['summary']['passed_triplets']} ({pass_rate*100:.1f}%)\n")
        f.write(f"Failed: {results['summary']['failed_triplets']} ({(1-pass_rate)*100:.1f}%)\n\n")
        
        # Detailed results
        f.write("="*70 + "\n")
        f.write("Detailed Results\n")
        f.write("="*70 + "\n\n")
        
        for triplet in results['triplet_results']:
            if 'error' in triplet:
                f.write(f"Triplet {triplet['triplet_number']}: ERROR - {triplet['error']}\n")
            else:
                status = "PASS" if triplet['passes_transitivity'] else "FAIL"
                f.write(f"Triplet {triplet['triplet_number']}: {status}\n")
                f.write(f"  Samples: {triplet['sample_a_id']}, {triplet['sample_b_id']}, {triplet['sample_c_id']}\n")
                f.write(f"  Results: AB={triplet['result_ab']}, BC={triplet['result_bc']}, AC={triplet['result_ac']}\n")
                f.write(f"  {triplet['explanation']}\n")
                f.write("\n")
    
    print(f"✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

