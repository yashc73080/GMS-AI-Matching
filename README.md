# GMS-AI-Matching

> **📋 Google Form**: [Submit your mentor/mentee application here](https://forms.gle/6spWTWthQi9G3vDaA)

A sophisticated mentor-mentee matching system that uses a combination of rule-based and semantic scoring to create optimal pairings based on preferences, compatibility, and goals.

## Overview

This project matches mentors and mentees using multiple scoring criteria including:
- Gender and ethnicity preferences
- MBTI personality compatibility
- Geographic proximity
- Academic year proximity
- Semantic similarity of goals and interests
- Rule-based goal alignment

## Features

- **Multiple Matching Algorithms**: Greedy assignment and Gale-Shapley stable matching
- **Capacity Management**: Respects mentor capacity constraints
- **Semantic Scoring**: Uses sentence transformers for goals and interests similarity
- **Google Sheets Integration**: Direct import from Google Forms responses
- **Detailed Analytics**: Comprehensive breakdown of matching criteria
- **Flexible Preferences**: Soft or hard enforcement of gender/ethnicity preferences

## Quick Start

### Installation

1. Clone the repository
```bash
git clone https://github.com/yashc73080/GMS-AI-Matching.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Set up your environment variables in `.env`:
```env
SHEET_ID='your_google_sheet_id'
GCRED_PATH='path_to_service_account.json'
```

The Service Account JSON file is from the Google Cloud Console, after enabling the Google Sheets API, creating a service account, and generating a JSON key. The Sheet ID is from the Google Sheets URL. Ensure the Google Sheet is shared to the client email found in the Service Account JSON. 

### Running the Matcher

```bash
python big_little.py
```

## Input Data

The system expects data with the following columns:
- `Name`, `Email`, `Mentee/Mentor`, `Gender`, `Ethnicity`, `State`, `School`, `Class Year`, `MBTI`, `Interests/Hobbies`, `Goals`, `Preferred Gender`, `Preferred Ethnicity`
- `Num Mentees` (for mentors only)

### Data Sources

- **CSV File**: Use [`sample_mentors_mentees.csv`](sample_mentors_mentees.csv)
- **Google Sheets**: Configure `SHEET_ID` and `GCRED_PATH` in your `.env`

## Configuration Options

Edit the configuration section in [`big_little.py`](big_little.py):

```python
INPUT_CSV = 'sample_mentors_mentees.csv'  # Or None for Google Sheets
SEMANTIC = 'embed'    # 'off' or 'embed' for semantic scoring
PREFS = 'soft'        # 'soft' or 'hard' preference enforcement
MATCH_ALGO = 'stable' # 'greedy' or 'stable' matching algorithm
```

## Scoring System

The matching score is calculated using weighted criteria:

| Criterion | Default Weight | Description |
|-----------|----------------|-------------|
| Gender Preference | 25 | Mentee's preferred gender matches mentor |
| Ethnicity Preference | 25 | Mentee's preferred ethnicity matches mentor |
| Goals (Semantic) | 25 | Semantic similarity of goals |
| Interests (Semantic) | 20 | Semantic similarity of interests |
| Year Proximity | 20 | Class years within 2 years |
| Location (State) | 15 | Same state |
| MBTI Compatibility | 15 | Based on compatibility matrix |
| Goals (Rule-based) | 5 | Keyword-based goal matching |

### MBTI Compatibility

Uses a predefined compatibility matrix in [`big_little.py`](big_little.py) that scores personality type compatibility from 0-15.

## Output Files

### `/out` Directory
- `mentor_mentee_full_scores.csv` - All mentor-mentee pair scores
- `mentor_mentee_topK_per_mentee.csv` - Top K matches per mentee
- `mentor_mentee_assigned_pairs_{algorithm}.csv` - Final assignments

### `/results` Directory
- [`mentor_mentee_assignments.csv`](results/mentor_mentee_assignments.csv) - Clean assignment list
- [`mentor_mentee_detailed.csv`](results/mentor_mentee_detailed.csv) - Detailed match breakdown

## Matching Algorithms

### Greedy Assignment
- [`greedy_assign_with_capacity`](matching_algos.py) - Assigns highest scoring pairs first
- Respects mentor capacity constraints
- Fast but may not be globally optimal

### Stable Matching (Gale-Shapley)
- [`gale_shapley_with_capacity`](matching_algos.py) - Ensures stable matches
- No mentor-mentee pair would prefer each other over current matches
- More computationally intensive but produces stable results

## Key Components

### Core Files
- [`big_little.py`](big_little.py) - Main script with scoring logic
- [`matching_algos.py`](matching_algos.py) - Matching algorithms
- [`semantic_scorer.py`](semantic_scorer.py) - Semantic similarity scoring
- [`sheet_helper.py`](sheet_helper.py) - Google Sheets integration and normalization

### Configuration
- [`requirements.txt`](requirements.txt) - Python dependencies
- [`.env`](.env) - Environment variables

## Semantic Scoring

When `SEMANTIC = 'embed'`, the system uses:
- **Model**: `all-MiniLM-L6-v2` (configurable)
- **Method**: Cosine similarity of sentence embeddings
- **Scope**: Goals and Interests/Hobbies comparison

To disable semantic scoring, set `SEMANTIC = 'off'` in [`big_little.py`](big_little.py).

## Understanding Results

The detailed output includes breakdown fields (`BRK::`) showing points awarded for each criterion. See [`brk_fields.md`](brk_fields.md) for detailed explanations.

## Google Forms Integration

Google Form Link: https://forms.gle/6spWTWthQi9G3vDaA

The system can directly import responses from Google Forms via Google Sheets API. Requires:
1. Service account JSON file
2. Sheet ID in environment variables
3. Proper API permissions

## Development

### Adding New Scoring Criteria
1. Update [`compute_score`](big_little.py) function
2. Add to `DEFAULT_WEIGHTS` dictionary
3. Include in breakdown (`brk`) dictionary

### Supporting New Input Fields
1. Update [`normalize_responses`](sheet_helper.py) for data cleaning
2. Add to expected columns list in [`big_little.py`](big_little.py)

## Dependencies

- `pandas` - Data manipulation
- `sentence-transformers` - Semantic similarity (optional)
- `scikit-learn` - Cosine similarity computation
- `gspread` - Google Sheets integration
- `numpy` - Numerical operations