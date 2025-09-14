from pathlib import Path
import re
import math
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from time import time

from sheet_helper import normalize_responses, read_google_sheet
from semantic_scorer import SemanticScorer
from matching_algos import greedy_assign, gale_shapley

# MBTI matrix; keep as-is for compatibility.
MBTI_MATRIX = {
    "INTJ": {"INTJ":8,"INTP":9,"ENTJ":10,"ENTP":11,"INFJ":9,"INFP":7,"ENFJ":10,"ENFP":15,
             "ISTJ":7,"ISFJ":6,"ESTJ":8,"ESFJ":7,"ISTP":6,"ISFP":5,"ESTP":7,"ESFP":6},
    "INTP": {"INTJ":9,"INTP":8,"ENTJ":9,"ENTP":10,"INFJ":8,"INFP":9,"ENFJ":7,"ENFP":8,
             "ISTJ":7,"ISFJ":6,"ESTJ":5,"ESFJ":4,"ISTP":7,"ISFP":8,"ESTP":6,"ESFP":5},
    "ENTJ": {"INTJ":10,"INTP":9,"ENTJ":8,"ENTP":10,"INFJ":12,"INFP":15,"ENFJ":11,"ENFP":9,
             "ISTJ":8,"ISFJ":7,"ESTJ":9,"ESFJ":8,"ISTP":7,"ISFP":3,"ESTP":8,"ESFP":6},
    "ENTP": {"INTJ":11,"INTP":10,"ENTJ":10,"ENTP":8,"INFJ":15,"INFP":11,"ENFJ":9,"ENFP":10,
             "ISTJ":7,"ISFJ":6,"ESTJ":8,"ESFJ":7,"ISTP":9,"ISFP":8,"ESTP":8,"ESFP":9},
    "INFJ": {"INTJ":9,"INTP":8,"ENTJ":12,"ENTP":15,"INFJ":8,"INFP":10,"ENFJ":11,"ENFP":12,
             "ISTJ":6,"ISFJ":8,"ESTJ":7,"ESFJ":9,"ISTP":7,"ISFP":9,"ESTP":4,"ESFP":6},
    "INFP": {"INTJ":7,"INTP":9,"ENTJ":15,"ENTP":11,"INFJ":10,"INFP":8,"ENFJ":15,"ENFP":11,
             "ISTJ":6,"ISFJ":7,"ESTJ":3,"ESFJ":6,"ISTP":7,"ISFP":8,"ESTP":6,"ESFP":7},
    "ENFJ": {"INTJ":10,"INTP":7,"ENTJ":11,"ENTP":9,"INFJ":11,"INFP":15,"ENFJ":8,"ENFP":12,
             "ISTJ":8,"ISFJ":9,"ESTJ":9,"ESFJ":10,"ISTP":4,"ISFP":6,"ESTP":7,"ESFP":8},
    "ENFP": {"INTJ":15,"INTP":8,"ENTJ":9,"ENTP":10,"INFJ":12,"INFP":11,"ENFJ":12,"ENFP":8,
             "ISTJ":3,"ISFJ":6,"ESTJ":5,"ESFJ":7,"ISTP":8,"ISFP":9,"ESTP":10,"ESFP":11},
    "ISTJ": {"INTJ":7,"INTP":7,"ENTJ":8,"ENTP":7,"INFJ":6,"INFP":6,"ENFJ":8,"ENFP":3,
             "ISTJ":8,"ISFJ":9,"ESTJ":9,"ESFJ":10,"ISTP":9,"ISFP":7,"ESTP":8,"ESFP":6},
    "ISFJ": {"INTJ":6,"INTP":6,"ENTJ":7,"ENTP":6,"INFJ":8,"INFP":7,"ENFJ":9,"ENFP":6,
             "ISTJ":9,"ISFJ":8,"ESTJ":10,"ESFJ":9,"ISTP":7,"ISFP":8,"ESTP":6,"ESFP":7},
    "ESTJ": {"INTJ":8,"INTP":5,"ENTJ":9,"ENTP":8,"INFJ":7,"INFP":3,"ENFJ":9,"ENFP":5,
             "ISTJ":9,"ISFJ":10,"ESTJ":8,"ESFJ":9,"ISTP":8,"ISFP":6,"ESTP":9,"ESFP":7},
    "ESFJ": {"INTJ":7,"INTP":4,"ENTJ":8,"ENTP":7,"INFJ":9,"INFP":6,"ENFJ":10,"ENFP":7,
             "ISTJ":10,"ISFJ":9,"ESTJ":9,"ESFJ":8,"ISTP":6,"ISFP":7,"ESTP":7,"ESFP":8},
    "ISTP": {"INTJ":6,"INTP":7,"ENTJ":7,"ENTP":9,"INFJ":7,"INFP":7,"ENFJ":4,"ENFP":8,
             "ISTJ":9,"ISFJ":7,"ESTJ":8,"ESFJ":6,"ISTP":8,"ISFP":9,"ESTP":10,"ESFP":9},
    "ISFP": {"INTJ":5,"INTP":8,"ENTJ":3,"ENTP":8,"INFJ":9,"INFP":8,"ENFJ":6,"ENFP":9,
             "ISTJ":7,"ISFJ":8,"ESTJ":6,"ESFJ":7,"ISTP":9,"ISFP":8,"ESTP":9,"ESFP":10},
    "ESTP": {"INTJ":7,"INTP":6,"ENTJ":8,"ENTP":8,"INFJ":4,"INFP":6,"ENFJ":7,"ENFP":10,
             "ISTJ":8,"ISFJ":6,"ESTJ":9,"ESFJ":7,"ISTP":10,"ISFP":9,"ESTP":8,"ESFP":9},
    "ESFP": {"INTJ":6,"INTP":5,"ENTJ":6,"ENTP":9,"INFJ":6,"INFP":7,"ENFJ":8,"ENFP":11,
             "ISTJ":6,"ISFJ":7,"ESTJ":7,"ESFJ":8,"ISTP":9,"ISFP":10,"ESTP":9,"ESFP":8}
}

# Default weights.
DEFAULT_WEIGHTS = {
    "gender_pref": 25,
    "eth_pref": 25,
    "year_proximity": 20,
    "mbti": 15,
    "goals_rule": 5,
    "goals_semantic": 25,
    "interests_semantic": 20,
    "location_state": 15,
}


def mbti_score_full(a: str, b: str) -> int:
    a = (a or "").upper().strip()
    b = (b or "").upper().strip()
    return MBTI_MATRIX.get(a, {}).get(b, 0)

def parse_list_cell(cell: Optional[str]):
    if pd.isna(cell) or cell is None:
        return set()
    items = re.split(r"[,/;]", str(cell))
    return set([it.strip().lower() for it in items if it.strip()])

def preference_satisfied(preferred, actual):
    if preferred is None:
        return True
    p = str(preferred).strip().lower()
    if p in ("", "no preference", "na", "n/a", "none"):
        return True
    return p == str(actual).strip().lower()

def goals_alignment_rules(mentee_goal, mentor_goal):
    mg = ("" if mentee_goal is None else str(mentee_goal)).strip().lower()
    pg = ("" if mentor_goal is None else str(mentor_goal)).strip().lower()
    if not mg or not pg:
        return 0
    if mg == pg:
        return 8
    related_sets = [
        {"network", "leadership"},
        {"research", "grad", "graduate", "masters", "phd"},
        {"business", "startup", "entrepreneurship"},
    ]
    tokens_m = set(re.findall(r"\w+", mg))
    tokens_p = set(re.findall(r"\w+", pg))
    for rs in related_sets:
        if tokens_m & rs and tokens_p & rs:
            return 4
    return 0

# Email validator for mentors: check .edu
def is_edu_email(email: Optional[str]) -> bool:
    if not email:
        return False
    return bool(re.search(r"@[^\s@]+\.edu$", str(email).strip().lower()))

# Scoring function
def compute_score(mentor, mentee, sem: SemanticScorer, weights: Dict[str, float], prefs_mode: str = "soft") -> Tuple[float, Dict[str, float]]:
    if str(mentor.get("Mentee/Mentor")).strip().lower() != "mentor" or str(mentee.get("Mentee/Mentor")).strip().lower() != "mentee":
        return -1.0, {}

    if prefs_mode == "hard":
        if not preference_satisfied(mentee.get("Preferred Gender"), mentor.get("Gender")):
            return -1.0, {}
        if not preference_satisfied(mentee.get("Preferred Ethnicity"), mentor.get("Ethnicity")):
            return -1.0, {}

    score = 0.0
    brk = {}

    # Preferred gender/ethnicity from mentee side
    gpref = weights.get("gender_pref", 0) if preference_satisfied(mentee.get("Preferred Gender"), mentor.get("Gender")) else 0
    epref = weights.get("eth_pref", 0) if preference_satisfied(mentee.get("Preferred Ethnicity"), mentor.get("Ethnicity")) else 0
    score += gpref + epref
    brk['pref_gender'] = gpref
    brk['pref_eth'] = epref

    # Mentor's preferences count half weight
    mgpref = 0.5 * weights.get("gender_pref", 0) if preference_satisfied(mentor.get("Preferred Gender"), mentee.get("Gender")) else 0
    mepref = 0.5 * weights.get("eth_pref", 0) if preference_satisfied(mentor.get("Preferred Ethnicity"), mentee.get("Ethnicity")) else 0
    score += mgpref + mepref
    brk['mentor_pref_gender'] = mgpref
    brk['mentor_pref_eth'] = mepref

    # Location match: state
    state_match = str(mentor.get("state","")).strip().lower() == str(mentee.get("state","")).strip().lower()
    loc_pts = weights.get("location_state", 0) if state_match else 0
    score += loc_pts
    brk['location_state'] = loc_pts

    # Year proximity
    yp = 0
    try:
        y1 = int(str(mentor.get("Class Year","")).strip())
        y2 = int(str(mentee.get("Class Year","")).strip())
        if abs(y1 - y2) <= 2:
            yp = weights.get("year_proximity", 0)
    except Exception:
        yp = 0
    score += yp
    brk['year_proximity'] = yp

    # MBTI
    mbti_pts = mbti_score_full(mentor.get("MBTI",""), mentee.get("MBTI",""))
    score += mbti_pts
    brk['mbti'] = mbti_pts

    # Goals rule-based
    gr = goals_alignment_rules(mentee.get("Goals"), mentor.get("Goals"))
    score += gr
    brk['goals_rule'] = gr

    # Semantic scoring if enabled
    gs01 = sem.goals(mentee.get("Goals"), mentor.get("Goals"))
    gs_pts = gs01 * weights.get("goals_semantic", 0)
    score += gs_pts
    brk['goals_semantic'] = round(gs_pts, 4)

    isem01 = sem.interests(mentee.get("Interest/Hobbies"), mentor.get("Interest/Hobbies"))
    isem_pts = isem01 * weights.get("interests_semantic", 0)
    score += isem_pts
    brk['interests_semantic'] = round(isem_pts, 4)

    return float(round(score, 4)), brk

# Build match table
def build_match_table(df: pd.DataFrame, sem: SemanticScorer, weights: Dict[str, float], prefs_mode: str = "soft"):
    # Basic validation and cleaning
    df = df.copy()
    # Normalize column names to expected ones if users exported from Google Forms
    # Expected fields provided by user: Class Year, Email, Gender, Ethnicity, state, Mentee/Mentor, Availability,
    # School, MBTI, Interest/Hobbies, Goals, Preferred Gender, Preferred Ethnicity, Name

    # Tag mentors without .edu for admin review
    df['mentor_email_is_edu'] = df['Email'].apply(is_edu_email)
    df['validation_notes'] = df.apply(lambda r: "" if not (str(r.get('Mentee/Mentor')).strip().lower()=='mentor' and not r['mentor_email_is_edu']) else 'Mentor email not .edu', axis=1)

    mentors = df[df['Mentee/Mentor'].str.strip().str.lower() == 'mentor'].reset_index(drop=True)
    mentees = df[df['Mentee/Mentor'].str.strip().str.lower() == 'mentee'].reset_index(drop=True)

    rows = []
    for _, me in mentees.iterrows():
        for _, mr in mentors.iterrows():
            s, br = compute_score(mr, me, sem=sem, weights=weights, prefs_mode=prefs_mode)
            if s >= 0:
                rows.append({
                    'Mentee': me.get('Name',''),
                    'Mentee Email': me.get('Email',''),
                    'Mentor': mr.get('Name',''),
                    'Mentor Email': mr.get('Email',''),
                    'Score': s,
                    'Mentor Email Is EDU': mr.get('mentor_email_is_edu', False),
                    'Validation Notes Mentor': mr.get('validation_notes',''),
                    'Shared Hobbies (Exact)': ', '.join(sorted(h_mentor & h_mentee for h_mentor, h_mentee in [(parse_list_cell(mr.get('Interest/Hobbies')), parse_list_cell(me.get('Interest/Hobbies')))] ) ) if False else ', '.join(sorted(parse_list_cell(mr.get('Interest/Hobbies')) & parse_list_cell(me.get('Interest/Hobbies')))),
                    'Same School': str(mr.get('School','')).strip().lower() == str(me.get('School','')).strip().lower(),
                    'Year Gap': (abs(int(mr.get('Class Year')) - int(me.get('Class Year'))) if str(mr.get('Class Year')).isdigit() and str(me.get('Class Year')).isdigit() else None),
                    'MBTI Mentee/Mentor': f"{mr.get('MBTI','')} / {me.get('MBTI','')}",
                    'Goals (Mentor←Mentee)': f"{mr.get('Goals','')} ← {me.get('Goals','')}",
                    **{f"BRK::{k}": v for k, v in br.items()}
                })
    scores_df = pd.DataFrame(rows).sort_values(['Mentee','Score'], ascending=[True, False]).reset_index(drop=True)
    return mentors, mentees, scores_df

# Write analysis tables
def write_analysis_tables(pairs_df: pd.DataFrame, df: pd.DataFrame, res_dir: Path):
    # Ensure output directory exists
    results_dir = res_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assignments Table
    assignments = pairs_df[["Mentor", "Mentor Email", "Mentee", "Mentee Email"]].copy()
    assignments.columns = ["mentor name", "mentor email", "mentee name", "mentee email"]
    assignments_path = results_dir / "mentor_mentee_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    # 2. Detailed Table
    # Fields used for matching (excluding Gender and Ethnicity)
    match_fields = [
        "state", "School", "Class Year", "MBTI", "Interest/Hobbies", "Goals", "Preferred Gender", "Preferred Ethnicity"
    ]
    detailed_rows = []
    for _, row in pairs_df.iterrows():
        mentor = df[df["Name"] == row["Mentor"]].iloc[0]
        mentee = df[df["Name"] == row["Mentee"]].iloc[0]
        detailed = {
            "mentor name": row["Mentor"],
            "mentee name": row["Mentee"]
        }
        # Add custom gender/ethnicity fields
        detailed["mentor Gender | mentee Preferred Gender"] = f"{mentor.get('Gender', '')} | {mentee.get('Preferred Gender', '')}"
        detailed["mentor Ethnicity | mentee Preferred Ethnicity"] = f"{mentor.get('Ethnicity', '')} | {mentee.get('Preferred Ethnicity', '')}"
        # Add all other match fields as "mentor_value | mentee_value" with column name "Field (mentor | mentee)"
        for field in match_fields:
            mentor_val = str(mentor.get(field, ""))
            mentee_val = str(mentee.get(field, ""))
            detailed[f"{field} (mentor | mentee)"] = f"{mentor_val} | {mentee_val}"
        # Add breakdown fields
        for brk in [
            "BRK::pref_gender", "BRK::pref_eth", "BRK::mentor_pref_gender", "BRK::mentor_pref_eth",
            "BRK::location_state", "BRK::year_proximity","BRK::mbti", "BRK::goals_rule", 
            "BRK::goals_semantic", "BRK::interests_semantic"
        ]:
            detailed[brk] = row.get(brk, "")
        detailed_rows.append(detailed)
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = results_dir / "mentor_mentee_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

# Main CLI
def main():
    start_time = time()

    # ---- BEGIN CONFIGURATION ----
    # Set these variables directly to configure the script
    INPUT_CSV = 'sample_mentors_mentees.csv'  # e.g., 'responses.csv'
    SHEET_ID = None   # e.g., 'your_google_sheet_id' or None to skip Google Sheets
    GCRED_PATH = None # e.g., 'service_account.json' if using Google Sheets
    OUTDIR = 'out'
    SEMANTIC = 'embed'  # 'off' or 'embed'
    EMBED_MODEL = 'all-MiniLM-L6-v2'
    PREFS = 'soft'    # 'soft' or 'hard';  how strictly to enforce gender and ethnicity preferences
    MATCH_ALGO = 'stable'  # 'greedy' or 'stable'
    TOPK = 3
    # ---- END CONFIGURATION ----

    out_dir = Path(OUTDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_dir = Path('results')
    res_dir.mkdir(parents=True, exist_ok=True)

    if SHEET_ID:
        if not GCRED_PATH:
            raise SystemExit('When using SHEET_ID you must provide GCRED_PATH to service account JSON')
        df = read_google_sheet(SHEET_ID, GCRED_PATH)
        df = normalize_responses(df)
        print(f"[INFO] Loaded {len(df)} rows from Google Sheet {SHEET_ID}")
    elif INPUT_CSV:
        df = pd.read_csv(INPUT_CSV)
        df = normalize_responses(df)
        print(f"[INFO] Loaded CSV {INPUT_CSV} ({len(df)} rows)")
    else:
        raise SystemExit('You must provide either INPUT_CSV or SHEET_ID to load data')

    # Ensure required columns
    expected = ["Name","Email","Mentee/Mentor","Gender","Ethnicity","state","School","Class Year","MBTI","Interest/Hobbies","Goals","Preferred Gender","Preferred Ethnicity"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns. Expected: {expected}. Missing: {missing}")

    sem = SemanticScorer(mode=SEMANTIC, model_name=EMBED_MODEL)

    mentors, mentees, scores_df = build_match_table(df, sem=sem, weights=DEFAULT_WEIGHTS, prefs_mode=PREFS)

    full_scores_path = out_dir / 'mentor_mentee_full_scores.csv'
    scores_df.to_csv(full_scores_path, index=False)

    topk = scores_df.groupby('Mentee').head(TOPK).reset_index(drop=True)
    topk_path = out_dir / f'mentor_mentee_top{TOPK}_per_mentee.csv'
    topk.to_csv(topk_path, index=False)

    if MATCH_ALGO == 'greedy':
        pairs = greedy_assign(scores_df)
    else:
        pairs = gale_shapley(scores_df)
    pairs_path = out_dir / f'mentor_mentee_assigned_pairs_{MATCH_ALGO}.csv'
    pairs.to_csv(pairs_path, index=False)

    write_analysis_tables(pairs, df, res_dir)

    end_time = time()

    print('[OK] Wrote:')
    print(f' - {full_scores_path}')
    print(f' - {topk_path}')
    print(f' - {pairs_path}')
    print(f' - {res_dir / "mentor_mentee_assignments.csv"}')
    print(f' - {res_dir / "mentor_mentee_detailed.csv"}')
    print(f'[INFO] Mentors: {len(mentors)}, Mentees: {len(mentees)}, Scored pairs: {len(scores_df)}')
    print(f'[INFO] Time elapsed: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    main()