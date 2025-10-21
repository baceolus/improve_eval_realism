# AI Conversation Realism Evaluator

A system for evaluating and ranking AI conversation transcripts by their realism using the Bradley-Terry rating model with LLM-based pairwise comparisons.

## Overview

This project implements a comprehensive evaluation framework to determine how realistic AI assistant conversations appear. It uses multiple LLM judges to perform pairwise comparisons between conversation transcripts, then applies the Bradley-Terry model to compute global ratings and rankings.

**Key Innovation**: Rather than relying on absolute scoring, this system uses relative comparisons to build a leaderboard, similar to how ELO ratings work in chess. 

## Features

### Core Functionality
- 🏆 **Bradley-Terry Leaderboard**: Calculate global ratings from pairwise comparisons using the MM algorithm
- 🔄 **Adaptive Integration**: Integrate new samples into existing leaderboards with minimal comparisons
- ⚡ **Parallel Processing**: High-throughput parallelized comparison execution
- 🎯 **Multiple Judge Prompts**: Uses 5 different judge prompts
- 📊 **Uncertainty Tracking**: Bayesian posterior variance updates for determing confidence intervals for rating values 

### Quality Assurance
- ✅ **Sanity Checks**: Automated validation of rating normalization, convergence, and other parameters
- 🔍 **Transitivity Testing**: Verify logical consistency of judge decisions
- 📏 **Order Bias Detection**: Test for positional bias in judge prompts (whether it predominantly selects the first sample it's presented or the seconf one)

## Setup

### Setup

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

4. Unizp datasets/processed_needham_dataset.json.zip
It's zipped in order to respect licensing requirements of some samples used for this dataset

### Required Packages
- `requests` - API communication
- `python-dotenv` - Environment variable management
- `networkx` - Graph analysis for diagnostics
- `numpy` - Numerical computations
- `inspect_ai` - For processing eval logs

## Quick Start

### 1. Calculate a Leaderboard from Scratch

```bash
python calculate_leaderboard.py
```

This will:
- Load samples from `datasets/processed_needham_dataset.json`
- Run adaptive pairwise comparisons (exploration then exploitation phases)
- Calculate Bradley-Terry ratings
- Save results to `leaderboards/bradley_terry_results_<timestamp>.json`

Configure parameters in the `main()` function:
```python
results = run_bradley_terry_competition(
    samples=agentic_samples,
    api_key=api_key,
    model="x-ai/grok-4-fast",
    max_rounds=16,              # Number of comparison rounds
    max_uncertainty_threshold=20.0,  # Stop when uncertainty is low enough
    max_workers=600             # Parallel workers
)
```

### 2. Integrate New Samples into Existing Leaderboard

```bash
python sample_integration.py <inspect_eval_file.eval> --model openai/gpt-5-mini
```

Example:
```bash
python sample_integration.py 2025-10-19T14-32-43+03-00_oversight-subversion.eval \
    --model openai/gpt-5-mini \
    --leaderboard leaderboards/gpt_5_mini_leaderboard.json \
    --output my_results
```

This will:
- Extract samples from the eval log
- Strategically compare them against calibration set samples
- Determine their leaderboard placement
- Save detailed results to `apollo_leaderboard_placement/`

## Project Structure

```
Improve eval realism/
├── README.md                          # This file
├── .env                               # API keys (not in repo)
│
├── Core Rating System
│   ├── bradley_terry.py               # Bradley-Terry MM algorithm implementation
│   ├── calculate_leaderboard.py      # Full leaderboard calculation pipeline
│   ├── sample_comparison.py          # LLM-based pairwise comparison logic
│   └── sample_integration.py         # Adaptive integration for new samples
│
├── Data Processing
│   ├── process_needham_dataset.py    # Dataset preprocessing utilities
│   └── extract_prompts_from_inspect_log.py  # Extract samples from eval logs
│
├── Quality Assurance
│   ├── basic_sanity_checks.py        # Automated validation checks
│   ├── diagnose_bradley_terry_convergence.py  # Convergence diagnostics
│   └── sanity_checks/
│       ├── transitivity_test.py      # Test logical consistency
│       └── order_consistency_test.py # Test for positional bias
│
├── Utilities
│   ├── aggregate_existing_results.py # Combine integration results
│   └── SAMPLE_INTEGRATION_GUIDE.md   # Detailed integration guide
│
└── Data & Results
    ├── datasets/                      # Input conversation datasets
    ├── leaderboards/                  # Calculated leaderboards
    ├── eval_logs/                     # Raw eval log files
    └── apollo_leaderboard_placement/  # Integration results
```

## How It Works

### 1. Pairwise Comparison with Multiple Judges

Each comparison between two transcripts uses **5 different judge prompts**, each applied in **both orders** (transcript A vs B, then B vs A), resulting in 10 independent judgments:

This design:
- Reduces bias from any single prompt
- Mitigates order effects
- Provides more robust decisions

### 2. Bradley-Terry Rating Model

The Bradley-Terry model estimates each sample's "strength" (θ) using the probability:

```
P(sample_i beats sample_j) = θ_i / (θ_i + θ_j)
```

We use the **Minorization-Maximization (MM) algorithm** to efficiently compute ratings from comparison results:


Ratings are on a log scale, centered at 0, with typical ranges of -500 to +500.

### 3. Uncertainty Estimation

Each rating has an associated uncertainty value (σ) computed using **Bayesian posterior variance updates**:
Uncertainty decreases as more comparisons are made, especially for close matchups.

### 4. Adaptive Integration Strategy

When adding new samples to an existing leaderboard:

1. **Initialize** at the median rating of the calibration set
2. **Find k-nearest neighbors** based on current rating estimate
3. **Select opponent** prioritizing high-confidence (low σ) calibration samples
4. **Run comparison** and update rating using Bradley-Terry
5. **Repeat** until uncertainty drops below threshold or max comparisons reached

This typically requires only 10-18 comparisons per sample vs. comparing against all calibration samples.

## Configuration

### Choosing a Model

Supported via OpenRouter:
- `openai/gpt-5-mini` - Fast, cost-effective
- `x-ai/grok-4-fast` - Alternative with different characteristics
- `anthropic/claude-sonnet-4.5` - High quality

### Key Parameters

**Leaderboard Calculation**:
- `max_rounds`: Number of comparison rounds (e.g., 16)
- `max_uncertainty_threshold`: Stop when max uncertainty drops below this (e.g., 20.0)
- `max_workers`: Parallel comparison workers (e.g., 600)

**Adaptive Integration**:
- `max_comparisons`: Maximum comparisons per new sample (e.g., 18)
- `uncertainty_threshold`: Target uncertainty for new samples (e.g., 20.0)
- `k_neighbors`: Candidate pool size for opponent selection (e.g., 10)

## Output Formats

### Leaderboard Results
```json
{
  "final_ratings": {
    "sample_id": {
      "rating": 125.43,
      "uncertainty": 18.32,
      "n_comparisons": 14
    }
  },
  "comparison_history": [...],
  "detailed_comparisons": [...],  // All judge decisions
  "summary_statistics": {...}
}
```

### Integration Results
```json
{
  "new_sample_id": "oversight_subversion:42",
  "final_theta": 87.65,
  "final_sigma": 19.12,
  "leaderboard_rank": 234,
  "leaderboard_percentile": 72.3,
  "comparisons": [...]  // Detailed comparison history
}
```