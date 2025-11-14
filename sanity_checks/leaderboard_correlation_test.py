"""
Leaderboard Correlation Sanity Check

This script compares two leaderboards by measuring the Spearman correlation
between the rankings of the same samples in both leaderboards.

It outputs:
- Spearman correlation coefficient and p-value
- Number of common samples
- Visual scatter plot showing the correlation between rankings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Add parent directory to path to import from other modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def load_leaderboard(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    Load a leaderboard JSON file.
    
    Args:
        filepath: Path to the leaderboard JSON file
        
    Returns:
        Dictionary mapping sample IDs to their ratings and stats
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('final_ratings', {})


def compute_rankings(ratings: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    """
    Convert ratings to rankings (1 = highest rating).
    
    Args:
        ratings: Dictionary mapping sample IDs to their rating info
        
    Returns:
        Dictionary mapping sample IDs to their rankings
    """
    # Sort by rating in descending order
    sorted_samples = sorted(
        ratings.items(),
        key=lambda x: x[1]['rating'],
        reverse=True
    )
    
    # Assign rankings (1-indexed)
    rankings = {}
    for rank, (sample_id, _) in enumerate(sorted_samples, start=1):
        rankings[sample_id] = rank
    
    return rankings


def get_common_samples(
    rankings1: Dict[str, int],
    rankings2: Dict[str, int]
) -> List[str]:
    """
    Find sample IDs that appear in both leaderboards.
    
    Args:
        rankings1: Rankings from first leaderboard
        rankings2: Rankings from second leaderboard
        
    Returns:
        List of common sample IDs
    """
    samples1 = set(rankings1.keys())
    samples2 = set(rankings2.keys())
    common = samples1.intersection(samples2)
    
    return sorted(common)


def compute_spearman_correlation(
    rankings1: Dict[str, int],
    rankings2: Dict[str, int],
    common_samples: List[str]
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between rankings.
    
    Args:
        rankings1: Rankings from first leaderboard
        rankings2: Rankings from second leaderboard
        common_samples: List of common sample IDs
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(common_samples) < 2:
        return 0.0, 1.0
    
    # Extract rankings for common samples
    ranks1 = [rankings1[sample_id] for sample_id in common_samples]
    ranks2 = [rankings2[sample_id] for sample_id in common_samples]
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(ranks1, ranks2)
    
    return correlation, p_value


def plot_ranking_correlation(
    rankings1: Dict[str, int],
    rankings2: Dict[str, int],
    common_samples: List[str],
    correlation: float,
    p_value: float,
    leaderboard1_name: str,
    leaderboard2_name: str
):
    """
    Create a scatter plot showing the correlation between rankings.
    
    Args:
        rankings1: Rankings from first leaderboard
        rankings2: Rankings from second leaderboard
        common_samples: List of common sample IDs
        correlation: Spearman correlation coefficient
        p_value: P-value of the correlation
        leaderboard1_name: Name of first leaderboard
        leaderboard2_name: Name of second leaderboard
    """
    # Extract rankings for common samples
    ranks1 = np.array([rankings1[sample_id] for sample_id in common_samples])
    ranks2 = np.array([rankings2[sample_id] for sample_id in common_samples])
    
    # Create figure with good size
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(ranks1, ranks2, alpha=0.5, s=30, color='#2E86AB', edgecolors='white', linewidths=0.5)
    
    # Add diagonal line (perfect correlation)
    max_rank = max(max(ranks1), max(ranks2))
    min_rank = min(min(ranks1), min(ranks2))
    plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', alpha=0.5, linewidth=2, label='Perfect correlation')
    
    # Labels and title
    plt.xlabel(f'Ranking in {leaderboard1_name}', fontsize=12, fontweight='bold')
    plt.ylabel(f'Ranking in {leaderboard2_name}', fontsize=12, fontweight='bold')
    plt.title(f'Leaderboard Ranking Correlation\nSpearman ρ = {correlation:.4f} (p = {p_value:.4e})',
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add text box with statistics
    textstr = f'Common samples: {len(common_samples)}\nSpearman ρ: {correlation:.4f}\np-value: {p_value:.4e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def compare_leaderboards(
    leaderboard1_path: str,
    leaderboard2_path: str,
    display_plot: bool = True
) -> Dict[str, any]:
    """
    Compare two leaderboards and compute Spearman correlation.
    
    Args:
        leaderboard1_path: Path to first leaderboard JSON
        leaderboard2_path: Path to second leaderboard JSON
        display_plot: Whether to display the correlation plot
        
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("Leaderboard Correlation Analysis")
    print("="*70)
    
    # Load leaderboards
    print(f"\nLoading leaderboards...")
    print(f"  Leaderboard 1: {leaderboard1_path}")
    print(f"  Leaderboard 2: {leaderboard2_path}")
    
    ratings1 = load_leaderboard(leaderboard1_path)
    ratings2 = load_leaderboard(leaderboard2_path)
    
    print(f"\n  Samples in leaderboard 1: {len(ratings1)}")
    print(f"  Samples in leaderboard 2: {len(ratings2)}")
    
    # Compute rankings
    print(f"\nComputing rankings...")
    rankings1 = compute_rankings(ratings1)
    rankings2 = compute_rankings(ratings2)
    
    # Find common samples
    common_samples = get_common_samples(rankings1, rankings2)
    print(f"\n  Common samples: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("\n❌ ERROR: No common samples found between leaderboards!")
        return {
            "error": "No common samples",
            "leaderboard1_samples": len(ratings1),
            "leaderboard2_samples": len(ratings2),
            "common_samples": 0
        }
    
    if len(common_samples) < 2:
        print("\n❌ ERROR: Not enough common samples to compute correlation!")
        return {
            "error": "Insufficient common samples",
            "leaderboard1_samples": len(ratings1),
            "leaderboard2_samples": len(ratings2),
            "common_samples": len(common_samples)
        }
    
    # Compute Spearman correlation
    print(f"\nComputing Spearman correlation...")
    correlation, p_value = compute_spearman_correlation(rankings1, rankings2, common_samples)
    
    # Print results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"\n  Spearman correlation (ρ): {correlation:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  Common samples: {len(common_samples)}")
    
    # Interpretation
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "marginally significant (p < 0.05)"
    else:
        significance = "not significant (p ≥ 0.05)"
    
    print(f"\n  Significance: {significance}")
    
    if abs(correlation) >= 0.9:
        strength = "very strong"
    elif abs(correlation) >= 0.7:
        strength = "strong"
    elif abs(correlation) >= 0.5:
        strength = "moderate"
    elif abs(correlation) >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"  Correlation strength: {strength}")
    
    # Sanity check assessment
    print("\n" + "="*70)
    if correlation >= 0.7 and p_value < 0.001:
        print("✓ PASS: Strong positive correlation detected")
        print("  The two leaderboards show consistent rankings.")
    elif correlation >= 0.5 and p_value < 0.01:
        print("⚠ CAUTION: Moderate correlation detected")
        print("  The leaderboards show some consistency but have notable differences.")
    else:
        print("❌ FAIL: Weak or no correlation detected")
        print("  The leaderboards show different rankings. This may indicate:")
        print("  - Different evaluation criteria")
        print("  - Different sample sets")
        print("  - Insufficient comparisons in one or both leaderboards")
    print("="*70 + "\n")
    
    # Display plot if requested
    if display_plot:
        leaderboard1_name = Path(leaderboard1_path).stem
        leaderboard2_name = Path(leaderboard2_path).stem
        plot_ranking_correlation(
            rankings1, rankings2, common_samples,
            correlation, p_value,
            leaderboard1_name, leaderboard2_name
        )
    
    # Return results
    return {
        "leaderboard1_path": leaderboard1_path,
        "leaderboard2_path": leaderboard2_path,
        "leaderboard1_samples": len(ratings1),
        "leaderboard2_samples": len(ratings2),
        "common_samples": len(common_samples),
        "spearman_correlation": correlation,
        "p_value": p_value,
        "significance": significance,
        "correlation_strength": strength
    }


def main():
    """
    Main function to run the leaderboard correlation test.
    """
    if len(sys.argv) < 3:
        print("Usage: python leaderboard_correlation_test.py <leaderboard1.json> <leaderboard2.json>")
        print("\nExample:")
        print("  python leaderboard_correlation_test.py leaderboards/grok_4_fast_leaderboard.json leaderboards/gpt_5_mini_leaderboard_v1.json")
        sys.exit(1)
    
    leaderboard1_path = sys.argv[1]
    leaderboard2_path = sys.argv[2]
    
    # Check if files exist
    if not Path(leaderboard1_path).exists():
        print(f"ERROR: File not found: {leaderboard1_path}")
        sys.exit(1)
    
    if not Path(leaderboard2_path).exists():
        print(f"ERROR: File not found: {leaderboard2_path}")
        sys.exit(1)
    
    # Run comparison
    results = compare_leaderboards(leaderboard1_path, leaderboard2_path, display_plot=True)
    
    # Check for errors
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()


