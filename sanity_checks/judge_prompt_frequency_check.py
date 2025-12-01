import json
import argparse
import os
from collections import defaultdict, Counter

def load_leaderboard(path):
    """Load the leaderboard JSON file."""
    print(f"Loading leaderboard from {path}...")
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def calculate_stats(detailed_comparisons):
    """
    Calculate agreement and bias statistics for each judge prompt.
    
    Returns:
        agreement_stats: dict mapping prompt_index -> {agreements, total}
        bias_stats: dict mapping prompt_index -> Counter(votes)
        position_stats: dict mapping prompt_index -> Counter(first/second/tie)
        decisive_count: number of comparisons with a clear winner (A or B)
    """
    # prompt_index -> {'agreements': count, 'total': count}
    # Tracks how often a prompt agrees with the final majority decision
    agreement_stats = defaultdict(lambda: {'agreements': 0, 'total': 0})
    
    # prompt_index -> Counter({'1': count, '2': count, 'tie': count})
    # Tracks the raw vote distribution for each prompt to detect bias between Sample A vs Sample B
    bias_stats = defaultdict(Counter)

    # prompt_index -> Counter({'first': count, 'second': count, 'tie': count})
    # Tracks preference for 1st vs 2nd position displayed (Position Bias)
    position_stats = defaultdict(Counter)
    
    decisive_count = 0

    for comp in detailed_comparisons:
        final_result = comp.get('final_result') # "A", "B", "tie", "error"
        
        # Determine the consensus winner code ('1' for A, '2' for B)
        consensus_winner = None
        if final_result == 'A':
            consensus_winner = '1'
        elif final_result == 'B':
            consensus_winner = '2'
            
        if consensus_winner:
            decisive_count += 1

        # Analyze individual judge votes within this comparison
        for res in comp.get('individual_results', []):
            prompt_idx = res.get('prompt_index')
            order = res.get('order', 'original')
            
            # Get the vote ("1", "2", "tie") - explicitly convert to string
            vote_data = res.get('result', {})
            if 'error' in vote_data: 
                continue
                
            vote = str(vote_data.get('more_realistic'))

            # 1. Track Bias (Sample A vs Sample B): Count every valid vote
            bias_stats[prompt_idx][vote] += 1
            
            # 2. Track Agreement: Only for comparisons with a clear winner
            if consensus_winner:
                agreement_stats[prompt_idx]['total'] += 1
                if vote == consensus_winner:
                    agreement_stats[prompt_idx]['agreements'] += 1

            # 3. Track Position Bias (First vs Second Displayed)
            # Logic:
            # order="original": displayed as (A, B)
            #   vote="1" (A) -> First
            #   vote="2" (B) -> Second
            # order="swapped": displayed as (B, A)
            #   vote="1" (A) -> Second (because A was displayed 2nd)
            #   vote="2" (B) -> First (because B was displayed 1st)
            
            pos_vote = None
            if vote == 'tie':
                pos_vote = 'tie'
            elif order == 'original':
                if vote == '1': pos_vote = 'first'
                elif vote == '2': pos_vote = 'second'
            elif order == 'swapped':
                if vote == '1': pos_vote = 'second' # A is 2nd
                elif vote == '2': pos_vote = 'first'  # B is 1st
            
            if pos_vote:
                position_stats[prompt_idx][pos_vote] += 1
                    
    return agreement_stats, bias_stats, position_stats, decisive_count

def print_report(agreement_stats, bias_stats, position_stats, decisive_count):
    """Print a formatted report of the statistics."""
    print(f"\nAnalyzed {decisive_count} decisive comparisons (where a winner was declared).")
    
    sorted_prompts = sorted(agreement_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    
    print("\n--- Agreement with Majority Decision ---")
    print(f"{'Prompt':<10} {'Agreement %':<15} {'Count':<10}")
    
    for idx in sorted_prompts:
        stats = agreement_stats[idx]
        total = stats['total']
        if total > 0:
            pct = (stats['agreements'] / total) * 100
            print(f"{idx:<10} {pct:.2f}%          {stats['agreements']}/{total}")
        else:
            print(f"{idx:<10} N/A              0/0")

    print("\n--- Judge Prompt Bias (Sample A vs Sample B) ---")
    print(f"{'Prompt':<10} {'Vote A %':<12} {'Vote B %':<12} {'Tie %':<12}")
    
    for idx in sorted_prompts:
        counts = bias_stats[idx]
        total = sum(counts.values())
        if total > 0:
            p1 = (counts['1'] / total) * 100
            p2 = (counts['2'] / total) * 100
            pt = (counts['tie'] / total) * 100
            print(f"{idx:<10} {p1:.1f}%        {p2:.1f}%        {pt:.1f}%")
        else:
             print(f"{idx:<10} N/A")

    print("\n--- Position Bias (First vs Second Displayed) ---")
    print(f"{'Prompt':<10} {'First %':<12} {'Second %':<12} {'Tie %':<12}")
    
    for idx in sorted_prompts:
        counts = position_stats[idx]
        total = sum(counts.values())
        if total > 0:
            p1 = (counts['first'] / total) * 100
            p2 = (counts['second'] / total) * 100
            pt = (counts['tie'] / total) * 100
            print(f"{idx:<10} {p1:.1f}%        {p2:.1f}%        {pt:.1f}%")
        else:
             print(f"{idx:<10} N/A")

def save_results(agreement_stats, bias_stats, position_stats, output_file):
    """Save results to JSON."""
    results = {
        'agreement_stats': dict(agreement_stats),
        'bias_stats': {k: dict(v) for k, v in bias_stats.items()},
        'position_stats': {k: dict(v) for k, v in position_stats.items()}
    }
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed stats saved to {output_file}")
    except Exception as e:
        print(f"Could not save JSON results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze judge prompt agreement in leaderboard.')
    parser.add_argument('leaderboard_path', nargs='?', default='leaderboards/grok_4_fast.json', 
                        help='Path to the leaderboard JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.leaderboard_path):
        print(f"File not found: {args.leaderboard_path}")
        return

    data = load_leaderboard(args.leaderboard_path)
    if not data:
        return

    detailed_comparisons = data.get('detailed_comparisons', [])
    print(f"Found {len(detailed_comparisons)} comparisons.")

    agreement_stats, bias_stats, position_stats, decisive_count = calculate_stats(detailed_comparisons)
    
    print_report(agreement_stats, bias_stats, position_stats, decisive_count)
    save_results(agreement_stats, bias_stats, position_stats, "sanity_checks/judge_prompt_stats.json")

if __name__ == "__main__":
    main()
