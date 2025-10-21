# Sample Integration Script - User Guide

## Overview

The `sample_integration.py` script implements an algorithm to integrate new samples into an existing Bradley-Terry rating system.

### What it does:
- Extracts samples from Inspect evaluation log files (`.eval` format)
- Runs pairwise comparisons against calibrated leaderboard samples
- Determines leaderboard placement with uncertainty estimates
- Supports parallel processing of multiple samples
- Saves comprehensive results and placement summaries

---

## Prerequisites

### Required Files
1. **Eval log file** (`.eval` format): Contains the new samples to be rated
2. **Leaderboard file** (`.json` format): Pre-calibrated Bradley-Terry leaderboard with existing sample ratings
3. **Dataset file** (`.json` format): Processed dataset containing all sample transcripts (used automatically)

### Required Dependencies
```bash
pip install python-dotenv
```

### Environment Variables
Create a `.env` file in your project root with:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

### Project Structure
```
igor_project/
├── sample_integration.py                # Main script
├── sample_comparison.py                 # Comparison utilities
├── extract_prompts_from_inspect_log.py  # Log extraction
├── bradley_terry.py                     # Bradley-Terry algorithm
├── eval_logs/                            
│   └── some_log.eval                    # Inspect eval log file
├── datasets/
│   └── processed_needham_dataset.json   # Sample transcripts
├── leaderboards/
│   └── gpt_5_mini_leaderboard.json      # Pre-calibrated leaderboard
└── apollo_leaderboard_placement/        # Output directory
```

---

## Usage


#### Basic Usage
```bash
python sample_integration.py <eval_file> --model <model_name> --leaderboard <leaderboard_file> --output ,output_file_name>
```
 
#### Required Arguments
- `eval_file`: Name or path of the `.eval` log file to process
- `--model` / `-m`: OpenRouter model to use for comparisons (e.g., `openai/gpt-5-mini`, `anthropic/claude-sonnet-4.5`)
- `--leaderboard`: Path to custom leaderboard JSON file (default: `leaderboards/gpt_5_mini_leaderboard.json`)

#### Optional Argument
- `--output` / `-o`: Base name for output files (default: auto-generated with timestamp)

#### Examples

**Example 1: Basic usage with filename only**
```bash
python3 sample_integration.py 2025-10-19T14-32-43+03-00_oversight-subversion_2s8d5NcbFfm9ffNrzJZyjt.eval --leaderboard leaderboards/gpt_5_mini_leaderboard.json --model openai/gpt-5-mini --output gpt_5_mini_test
```

## Algorithm Overview

### Adaptive Integration Process

For each new sample, the algorithm:

1. **Initialization**
   - Starts with rating estimate at median of existing samples
   - Initial uncertainty: σ = 100.0

2. **Iterative Refinement** (up to 14 comparisons by default)
   - Find k=10 nearest neighbors to current rating estimate
   - Select best opponent (balancing confidence and proximity)
   - Run pairwise comparison using specified model
   - Update rating using Bradley-Terry MM on all comparisons so far
   - Recalculate uncertainty: σ = 100 / √(n_comparisons)

3. **Convergence**
   - Stops when uncertainty σ < 20.0 (target threshold)
   - Or when max comparisons (14) reached

4. **Placement**
   - Calculates rank, percentile on leaderboard
   - Saves detailed results

### Opponent Selection Strategy
- Prioritizes samples with **low uncertainty** (high confidence)
- Prefers samples with ratings **close to current estimate**
- Uses 2:1 weighting ratio for confidence vs. proximity

### Rating Calculation
- Uses Bradley-Terry MM algorithm (same as initial leaderboard)
- Anchors opponent ratings to leaderboard values (prevents drift)
- Adjusts for any shifts in opponent ratings
- Calculates uncertainty based on number of comparisons

---

## Output Files

All results are saved to `apollo_leaderboard_placement/` directory.

### 1. Aggregate Results JSON
**Filename**: `aggregate_integration_results_<timestamp>.json` or `<output_name>.json`

**Structure**:
```json
{
  "metadata": {
    "timestamp": "2025-10-19_15-05-52",
    "total_samples": 5,
    "total_comparisons": 65,
    "average_comparisons": 13.0,
    "average_theta": 45.2,
    "average_sigma": 20.5,
    "convergence_statistics": {
      "reached_threshold": 4,
      "hit_max_comparisons": 1,
      "convergence_rate": "80.0%"
    },
    "leaderboard_statistics": {
      "leaderboard_size": 100,
      "best_rank": 15,
      "worst_rank": 87,
      "median_rank": 42
    }
  },
  "samples": [
    {
      "new_sample_id": "sample_001",
      "final_theta": 52.3,
      "final_sigma": 19.8,
      "initial_theta": 50.0,
      "initial_sigma": 100.0,
      "num_comparisons": 14,
      "convergence_reason": "Uncertainty threshold reached",
      "leaderboard_rank": 25,
      "leaderboard_total": 100,
      "leaderboard_percentile": 75.0,
      "comparisons": [
        {
          "comparison_num": 1,
          "opponent_id": "calibration_sample_42",
          "opponent_theta": 48.5,
          "opponent_sigma": 12.3,
          "result": 1.0,
          "theta_before": 50.0,
          "sigma_before": 100.0,
          "theta_after": 53.2,
          "sigma_after": 100.0
        }
        // ... more comparisons
      ]
    }
    // ... more samples
  ]
}
```

### 2. Placement Summary Text
**Filename**: `placement_summary_<timestamp>.txt` or `<output_name>_summary.txt`

**Example**:
```
================================================================================
LEADERBOARD PLACEMENT SUMMARY
================================================================================
Generated: 2025-10-19_15-05-52
Total samples processed: 5
Total comparisons: 65
Average theta: 45.23
Leaderboard size: 100 samples

================================================================================
PLACEMENTS
================================================================================
Rank       Sample ID                                          Rating               Percentile  
--------------------------------------------------------------------------------
#15/100    sample_003                                         62.34 ± 18.45        85th        
#25/100    sample_001                                         52.31 ± 19.82        75th        
#42/100    sample_005                                         43.22 ± 20.12        58th        
#63/100    sample_002                                         35.67 ± 21.34        37th        
#87/100    sample_004                                         28.91 ± 22.01        13th        
```