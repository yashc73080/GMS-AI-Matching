import os
from pathlib import Path
import re
from typing import Optional, Dict, Tuple
import pandas as pd
from time import time

from sheet_helper import normalize_responses, read_google_sheet
from semantic_scorer import SemanticScorer
from matching_algos import greedy_assign, gale_shapley, greedy_assign_with_capacity, gale_shapley_with_capacity

from dotenv import load_dotenv
load_dotenv()

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
    "gender_pref": 20,
    "eth_pref": 20,
    "lgbtq_pref": 20,
    "availability_overlap": 20,
    "mbti": 20,
    "goals_semantic": 30,
    "interests_semantic": 25,
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
        return False
    p = str(preferred).strip().lower()
    if p in ("", "no preference", "na", "n/a", "none"):
        return True
    return p == str(actual).strip().lower()

def check_preference_match(preference_answer: Optional[str], mentor_value, mentee_value) -> bool:
    """
    Check if a preference is satisfied based on the new schema format.
    preference_answer: "Yes", "No", or "No Preference"
    Returns True if the preference is satisfied or if it's "No Preference"
    """
    if preference_answer is None:
        return True
    pref = str(preference_answer).strip().lower()
    if pref in ("no", "no preference", "na", "n/a", "none", ""):
        return True
    if pref == "yes":
        # For "Yes", check if values match
        mentor_val = str(mentor_value).strip().lower() if mentor_value else ""
        mentee_val = str(mentee_value).strip().lower() if mentee_value else ""
        return mentor_val == mentee_val and mentor_val != ""
    return True

def calculate_availability_overlap(mentor_avail: Dict, mentee_avail: Dict) -> float:
    """
    Calculate the overlap in availability between mentor and mentee.
    Returns a score between 0 and 1 based on matching time slots.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    total_slots = 0
    matching_slots = 0
    
    for day in days:
        mentor_day_col = f"Availability [{day}]"
        mentee_day_col = f"Availability [{day}]"
        
        mentor_times = parse_list_cell(mentor_avail.get(mentor_day_col))
        mentee_times = parse_list_cell(mentee_avail.get(mentee_day_col))
        
        if mentor_times or mentee_times:
            total_slots += 1
            if mentor_times and mentee_times:
                overlap = mentor_times.intersection(mentee_times)
                if overlap:
                    matching_slots += len(overlap) / max(len(mentor_times), len(mentee_times))
    
    if total_slots == 0:
        return 0.0
    return matching_slots / total_slots

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

    score = 0.0
    brk = {}

    # NEW SCHEMA: Gender preference (mentee side)
    # "Would you like to be matched with someone of similar gender?"
    mentee_wants_similar_gender = check_preference_match(
        mentee.get("Prefer Similar Gender"),
        mentor.get("Gender"),
        mentee.get("Gender")
    )
    gpref = weights.get("gender_pref", 0) if mentee_wants_similar_gender else 0
    
    # Hard mode check for gender preference
    if prefs_mode == "hard" and not mentee_wants_similar_gender:
        return -1.0, {}
    
    score += gpref
    brk['pref_gender'] = gpref

    # NEW SCHEMA: Ethnicity preference (mentee side)
    # "Would you like to be matched with someone of similar ethnic demographics?"
    mentee_wants_similar_ethnicity = check_preference_match(
        mentee.get("Prefer Similar Ethnicity"),
        mentor.get("Ethnicity"),
        mentee.get("Ethnicity")
    )
    epref = weights.get("eth_pref", 0) if mentee_wants_similar_ethnicity else 0
    
    # Hard mode check for ethnicity preference
    if prefs_mode == "hard" and not mentee_wants_similar_ethnicity:
        return -1.0, {}
    
    score += epref
    brk['pref_eth'] = epref

    # Mentor's preferences count half weight
    mentor_wants_similar_gender = check_preference_match(
        mentor.get("Prefer Similar Gender"),
        mentee.get("Gender"),
        mentor.get("Gender")
    )
    mgpref = 0.5 * weights.get("gender_pref", 0) if mentor_wants_similar_gender else 0
    score += mgpref
    brk['mentor_pref_gender'] = mgpref

    mentor_wants_similar_ethnicity = check_preference_match(
        mentor.get("Prefer Similar Ethnicity"),
        mentee.get("Ethnicity"),
        mentor.get("Ethnicity")
    )
    mepref = 0.5 * weights.get("eth_pref", 0) if mentor_wants_similar_ethnicity else 0
    score += mepref
    brk['mentor_pref_eth'] = mepref

    # NEW SCHEMA: LGBTQ+ preference matching
    # "Would you like to be matched with someone from the LGBTQ+ community?"
    lgbtq_pts = 0
    mentee_lgbtq_pref = str(mentee.get("Prefer LGBTQ+", "")).strip().lower()
    mentor_lgbtq_status = str(mentor.get("LGBTQ+", "")).strip().lower()
    
    if mentee_lgbtq_pref == "yes" and mentor_lgbtq_status == "yes":
        lgbtq_pts = weights.get("lgbtq_pref", 0)
    elif mentee_lgbtq_pref in ("no preference", "no", ""):
        lgbtq_pts = weights.get("lgbtq_pref", 0) * 0.5  # Partial credit for no preference
    
    # Check mentor's LGBTQ+ preference too (half weight)
    mentor_lgbtq_pref = str(mentor.get("Prefer LGBTQ+", "")).strip().lower()
    mentee_lgbtq_status = str(mentee.get("LGBTQ+", "")).strip().lower()
    
    if mentor_lgbtq_pref == "yes" and mentee_lgbtq_status == "yes":
        lgbtq_pts += 0.5 * weights.get("lgbtq_pref", 0)
    elif mentor_lgbtq_pref in ("no preference", "no", ""):
        lgbtq_pts += 0.25 * weights.get("lgbtq_pref", 0)
    
    score += lgbtq_pts
    brk['lgbtq_pref'] = round(lgbtq_pts, 4)

    # NEW SCHEMA: Availability overlap
    avail_overlap = calculate_availability_overlap(mentor, mentee)
    avail_pts = avail_overlap * weights.get("availability_overlap", 0)
    score += avail_pts
    brk['availability_overlap'] = round(avail_pts, 4)

    # Location match: state
    state_match = str(mentor.get("State","")).strip().lower() == str(mentee.get("State","")).strip().lower()
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

    isem01 = sem.interests(mentee.get("Interests/Hobbies"), mentor.get("Interests/Hobbies"))
    isem_pts = isem01 * weights.get("interests_semantic", 0)
    score += isem_pts
    brk['interests_semantic'] = round(isem_pts, 4)

    return float(round(score, 4)), brk

# Build match table
def build_match_table(df: pd.DataFrame, sem: SemanticScorer, weights: Dict[str, float], prefs_mode: str = "soft"):
    # Basic validation and cleaning
    df = df.copy()
    # Normalize column names to expected ones if users exported from Google Forms
    # Expected fields: Full Name, Email, Gender, Ethnicity, State, Mentee/Mentor, LGBTQ+,
    # Availability [Monday-Sunday], School, Class Year, MBTI, Interests/Hobbies, Goals,
    # Prefer Similar Gender, Prefer Similar Ethnicity, Prefer LGBTQ+, Num Mentees

    # Tag mentors without .edu for admin review
    df['mentor_email_is_edu'] = df['Email'].apply(is_edu_email)
    
    # Build validation notes
    def build_validation_notes(row):
        notes = []
        role = str(row.get('Mentee/Mentor', '')).strip().lower()
        
        if role == 'mentor':
            if not row['mentor_email_is_edu']:
                notes.append('Mentor email not .edu')
            # Check if mentor has University filled (not High School)
            if pd.isna(row.get('School')) or not row.get('School'):
                notes.append('Mentor missing University')
            # Note: 'School' column should be from 'University' after normalization
        elif role == 'mentee':
            # Check if mentee has High School filled (not University)
            if pd.isna(row.get('School')) or not row.get('School'):
                notes.append('Mentee missing High School')
            # Note: 'School' column should be from 'High School' after normalization
        
        return '; '.join(notes) if notes else ''
    
    df['validation_notes'] = df.apply(build_validation_notes, axis=1)

    # Separate mentors and mentees - only match mentor to mentee, never mentee to mentee
    mentors = df[df['Mentee/Mentor'].str.strip().str.lower() == 'mentor'].reset_index(drop=True)
    mentees = df[df['Mentee/Mentor'].str.strip().str.lower() == 'mentee'].reset_index(drop=True)
    
    # Log validation warnings
    invalid_mentors = mentors[mentors['validation_notes'] != '']
    invalid_mentees = mentees[mentees['validation_notes'] != '']
    
    if len(invalid_mentors) > 0:
        print(f"[WARN] {len(invalid_mentors)} mentors have validation issues:")
        for _, m in invalid_mentors.iterrows():
            print(f"  - {m.get('Name')}: {m['validation_notes']}")
    
    if len(invalid_mentees) > 0:
        print(f"[WARN] {len(invalid_mentees)} mentees have validation issues:")
        for _, m in invalid_mentees.iterrows():
            print(f"  - {m.get('Name')}: {m['validation_notes']}")

    rows = []
    # Only iterate mentor x mentee (never mentee x mentee or mentor x mentor)
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
                    'Validation Notes Mentee': me.get('validation_notes',''),
                    'Shared Hobbies (Exact)': ', '.join(sorted(h_mentor & h_mentee for h_mentor, h_mentee in [(parse_list_cell(mr.get('Interests/Hobbies')), parse_list_cell(me.get('Interests/Hobbies')))] ) ) if False else ', '.join(sorted(parse_list_cell(mr.get('Interests/Hobbies')) & parse_list_cell(me.get('Interests/Hobbies')))),
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

    # Get requested mentees from responses
    mentor_requested = {}
    if "Num Mentees" in df.columns:
        for _, row in df[df["Mentee/Mentor"].str.strip().str.lower() == "mentor"].iterrows():
            mentor_requested[row["Name"]] = row.get("Num Mentees", "")

    # Count allocated mentees from pairs
    mentor_allocated = pairs_df["Mentor"].value_counts().to_dict()

    # 2. Detailed Table
    match_fields = [
        "State", "School", "Class Year", "MBTI", "Interests/Hobbies", "Goals", "Preferred Gender", "Preferred Ethnicity"
    ]
    detailed_rows = []
    for _, row in pairs_df.iterrows():
        mentor = df[df["Name"] == row["Mentor"]].iloc[0]
        mentee = df[df["Name"] == row["Mentee"]].iloc[0]
        detailed = {
            "mentor name": row["Mentor"],
            "mentee name": row["Mentee"],
            "Num Mentees Requested": mentor_requested.get(row["Mentor"], ""),
            "Num Mentees Allocated": mentor_allocated.get(row["Mentor"], 0)
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
    INPUT_CSV = None # 'sample_mentors_mentees.csv'  # None to use Google Sheets
    SHEET_ID = os.getenv("SHEET_ID") # None to skip Google Sheets
    GCRED_PATH = os.getenv("GCRED_PATH") # None to skip Google Sheets
    OUTDIR = 'out'
    SEMANTIC = 'embed'  # 'off' or 'embed'
    EMBED_MODEL = 'all-MiniLM-L6-v2'
    PREFS = 'soft'    # 'soft' or 'hard'; how strictly to enforce gender and ethnicity preferences
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
        df.to_csv(out_dir/'google_sheet_responses.csv', index=False)
    elif INPUT_CSV:
        df = pd.read_csv(INPUT_CSV)
        df = normalize_responses(df)
        print(f"[INFO] Loaded CSV {INPUT_CSV} ({len(df)} rows)")
    else:
        raise SystemExit('You must provide either INPUT_CSV or SHEET_ID to load data')

    # Ensure required columns (updated for new schema)
    expected = ["Name","Email","Mentee/Mentor","Gender","LGBTQ+","Ethnicity","State","School","Class Year","MBTI","Interests/Hobbies","Goals","Prefer Similar Gender","Prefer Similar Ethnicity","Prefer LGBTQ+"]
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
        pairs = greedy_assign_with_capacity(scores_df, mentors)
    else:
        pairs = gale_shapley_with_capacity(scores_df, mentors)

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