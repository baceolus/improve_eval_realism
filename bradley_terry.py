"""
Bradley-Terry rating system implementation using the Minorization-Maximization (MM) algorithm.

This module provides functions for:
- Calculating Bradley-Terry ratings from pairwise comparisons
- Estimating uncertainty for each rating
- Supporting ties in comparisons
"""

import math
from typing import Dict, List, Tuple, Any


# Initial uncertainty value for all samples before any comparisons
INITIAL_SIGMA = 250.0


def bradley_terry_mm(
    comparisons: List[Tuple[str, str, str]],
    max_iterations: int = 500,
    tolerance: float = 0.01,
    lambda_reg: float = 0.5
) -> Dict[str, float]:
    """
    Estimate Bradley-Terry ratings using the Minorization-Maximization algorithm.
    
    The Bradley-Terry model estimates the probability that sample i beats sample j as:
    P(i > j) = θ_i / (θ_i + θ_j)
    
    where θ_i is the "strength" or rating of sample i.
    
    Args:
        comparisons: List of (sample_i, sample_j, result) tuples where:
            - sample_i, sample_j are sample identifiers
            - result is "A" (i wins), "B" (j wins), or "tie"
        max_iterations: Maximum number of MM iterations
        tolerance: Convergence threshold for max rating change
        lambda_reg: Regularization parameter (default: 0.5). Prevents ratings from
            diverging to infinity for samples with perfect win/loss records.
            Higher values = stronger regularization = faster convergence but less
            extreme ratings. Set to 0 to disable regularization.
    
    Returns:
        Dictionary mapping sample IDs to their ratings (log scale, centered around 0)
    """
    if not comparisons:
        return {}
    
    # Extract all unique samples
    all_samples = set()
    for sample_i, sample_j, _ in comparisons:
        all_samples.add(sample_i)
        all_samples.add(sample_j)
    all_samples = list(all_samples)
    
    # Initialize all ratings to 1.0
    ratings = {sample_id: 1.0 for sample_id in all_samples}
    
    # MM iterations
    for iteration in range(max_iterations):
        old_ratings = ratings.copy()
        
        # Update step for each sample
        for sample_i in all_samples:
            # Count wins for sample i
            wins_i = sum(
                1 for (a, b, result) in comparisons
                if (a == sample_i and result == "A") or (b == sample_i and result == "B")
            )
            
            # Count ties for sample i (count as 0.5 wins)
            ties_i = sum(
                1 for (a, b, result) in comparisons
                if (a == sample_i or b == sample_i) and result == "tie"
            )
            
            # Sum of 1/(θ_i + θ_j) over all comparisons involving i
            # Use old_ratings for denominator (correct MM algorithm)
            # Add regularization term to prevent divergence
            denominator = lambda_reg
            for sample_a, sample_b, _ in comparisons:
                if sample_a == sample_i:
                    denominator += 1.0 / (old_ratings[sample_i] + old_ratings[sample_b])
                elif sample_b == sample_i:
                    denominator += 1.0 / (old_ratings[sample_i] + old_ratings[sample_a])
            
            # MM update formula with regularization
            # Regularization adds virtual wins and comparisons to stabilize extreme cases
            if denominator > 0:
                ratings[sample_i] = (wins_i + 0.5 * ties_i + lambda_reg) / denominator
            else:
                ratings[sample_i] = 1.0
        
        # Normalize ratings (for numerical stability)
        mean_rating = sum(ratings.values()) / len(ratings) if ratings else 1.0
        if mean_rating > 0:
            # Normalize old_ratings too for fair comparison
            old_mean = sum(old_ratings.values()) / len(old_ratings) if old_ratings else 1.0
            old_ratings_normalized = {k: v / old_mean for k, v in old_ratings.items()}
            ratings = {k: v / mean_rating for k, v in ratings.items()}
        else:
            old_ratings_normalized = old_ratings
        
        # Check convergence (compare normalized values)
        max_change = max(abs(ratings[k] - old_ratings_normalized[k]) for k in ratings) if ratings else 0
        if max_change < tolerance:
            break
    
    # Convert to log scale and center around 0
    log_ratings = {}
    for sample_id, rating in ratings.items():
        if rating > 0:
            log_ratings[sample_id] = math.log(rating) * 100
        else:
            log_ratings[sample_id] = -1000  # Handle edge case
    
    # Center ratings around 0
    mean_log_rating = sum(log_ratings.values()) / len(log_ratings) if log_ratings else 0
    centered_ratings = {k: v - mean_log_rating for k, v in log_ratings.items()}
    
    return centered_ratings


def calculate_uncertainties(
    ratings: Dict[str, float],
    comparisons: List[Tuple[str, str, str]],
    initial_sigma: float = INITIAL_SIGMA
) -> Dict[str, float]:
    """
    Calculate uncertainty estimates using Bayesian posterior variance updates.
    
    Sigma (uncertainty) is updated after each comparison using Fisher information:
    - sigma_new = 1 / sqrt(1/sigma^2 + I(result))
    - I(result) = p_A * (1 - p_A) where p_A is the predicted win probability
    
    This makes sigma shrink faster for informative comparisons (close matchups)
    and slower for uninformative comparisons (mismatches).
    
    Args:
        ratings: Dictionary of sample ratings (on log scale, centered at 0)
        comparisons: List of (sample_i, sample_j, result) tuples
        initial_sigma: Initial uncertainty value (default: INITIAL_SIGMA)
    
    Returns:
        Dictionary mapping sample IDs to their uncertainty values
    """
    # Initialize all uncertainties to high value
    uncertainties = {sample_id: initial_sigma for sample_id in ratings.keys()}
    
    # Process each comparison in order to update uncertainties
    for sample_a, sample_b, result in comparisons:
        theta_a = ratings.get(sample_a, 0.0)
        theta_b = ratings.get(sample_b, 0.0)
        sigma_a = uncertainties.get(sample_a, initial_sigma)
        sigma_b = uncertainties.get(sample_b, initial_sigma)
        
        # Calculate predicted win probability for A
        # p_A = 1 / (1 + exp(theta_B - theta_A))
        # Ratings are on log scale (multiplied by 100), so divide by 100
        theta_diff = (theta_b - theta_a) / 100.0
        
        # Use stable computation to avoid overflow
        if theta_diff > 20:
            p_a = 0.0
        elif theta_diff < -20:
            p_a = 1.0
        else:
            p_a = 1.0 / (1.0 + math.exp(theta_diff))
        
        # Calculate Fisher information
        # I = p_A * (1 - p_A)
        # Scale by 100^2 to match the rating scale (ratings are log(strength) * 100)
        fisher_info = p_a * (1.0 - p_a) / (100.0 ** 2)
        
        # Update sigma for both samples using Bayesian posterior variance
        # sigma_new = 1 / sqrt(1/sigma^2 + I)
        if fisher_info > 0:
            precision_a = 1.0 / (sigma_a ** 2)
            precision_b = 1.0 / (sigma_b ** 2)
            
            new_precision_a = precision_a + fisher_info
            new_precision_b = precision_b + fisher_info
            
            uncertainties[sample_a] = 1.0 / math.sqrt(new_precision_a)
            uncertainties[sample_b] = 1.0 / math.sqrt(new_precision_b)
    
    return uncertainties


def get_comparison_counts(comparisons: List[Tuple[str, str, str]]) -> Dict[str, int]:
    """
    Count the number of comparisons for each sample.
    
    Args:
        comparisons: List of (sample_i, sample_j, result) tuples
    
    Returns:
        Dictionary mapping sample IDs to their comparison counts
    """
    counts = {}
    
    for sample_a, sample_b, _ in comparisons:
        counts[sample_a] = counts.get(sample_a, 0) + 1
        counts[sample_b] = counts.get(sample_b, 0) + 1
    
    return counts


def get_rating_statistics(ratings: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a set of ratings.
    
    Args:
        ratings: Dictionary of sample ratings
    
    Returns:
        Dictionary with min, max, median, Q1, Q3 statistics
    """
    if not ratings:
        return {
            "min": 0,
            "max": 0,
            "median": 0,
            "q1": 0,
            "q3": 0,
            "mean": 0,
            "std": 0
        }
    
    sorted_ratings = sorted(ratings.values())
    n = len(sorted_ratings)
    
    def percentile(data, p):
        """Calculate the p-th percentile of data."""
        if not data:
            return 0
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1
    
    mean = sum(sorted_ratings) / n
    variance = sum((x - mean) ** 2 for x in sorted_ratings) / n
    std = math.sqrt(variance)
    
    return {
        "min": sorted_ratings[0],
        "max": sorted_ratings[-1],
        "median": percentile(sorted_ratings, 0.5),
        "q1": percentile(sorted_ratings, 0.25),
        "q3": percentile(sorted_ratings, 0.75),
        "mean": mean,
        "std": std
    }


def get_uncertainty_statistics(uncertainties: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a set of uncertainties.
    
    Args:
        uncertainties: Dictionary of sample uncertainties
    
    Returns:
        Dictionary with min, max, median, mean statistics
    """
    return get_rating_statistics(uncertainties)

