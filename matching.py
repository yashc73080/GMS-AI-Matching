"""
Updated mentor_mentee_matcher.py

Features added and simplified per user request:
- Reads responses from CSV or Google Sheets (optional, via gspread + service account)
- Validates mentor .edu emails and tags non-.edu mentors for review
- Splits data into mentors and mentees dataframes
- Matching algorithm that considers:
  - MBTI compatibility via the existing MBTI matrix
  - Preferred gender and ethnicity (soft or hard preferences)
  - Location match (state level) with configurable weight
  - Class year proximity
  - Exact hobby overlap
  - Semantic similarity of Goals and Interests using sentence-transformers embeddings (optional)
- Produces: full score CSV, top-K per mentee CSV, and assigned pairs CSV
- Admin review flags for any validation issues

Usage:
  python mentor_mentee_matcher_updated.py --input responses.csv --outdir out --semantic embed

Dependencies (optional):
  pip install pandas numpy scikit-learn sentence-transformers gspread oauth2client

"""

from pathlib import Path
import argparse
import re
import math
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from google.oauth2.service_account import Credentials
from time import time

# MBTI matrix retained from original file. Keep as-is for compatibility.
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


def mbti_score_full(a: str, b: str) -> int:
    a = (a or "").upper().strip()
    b = (b or "").upper().strip()
    return MBTI_MATRIX.get(a, {}).get(b, 0)

# Default weights. Includes location weight for state matching.
DEFAULT_WEIGHTS = {
    "gender_pref": 30,
    "eth_pref": 30,
    "same_school": 20,
    "year_proximity": 20,
    "hobby_overlap": 10,
    "mbti": 15,
    "goals_rule": 8,
    "goals_semantic": 20,
    "interests_semantic": 10,
    "location_state": 15,
}

# Semantic scorer simplified and optional
class SemanticScorer:
    def __init__(self, mode: str = "off", model_name: str = "all-MiniLM-L6-v2"):
        self.mode = mode
        self.model_name = model_name
        self._embedder = None
        if self.mode == "embed":
            try:
                self._embedder = SentenceTransformer(self.model_name)
                self.cosine_similarity = cosine_similarity
            except Exception as e:
                raise RuntimeError("Embed mode requires sentence-transformers and scikit-learn. "
                                   "Install with: pip install sentence-transformers scikit-learn")

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return "" if s is None else str(s).strip()

    def _embed_cosine01(self, a: str, b: str) -> float:
        a = self._norm(a); b = self._norm(b)
        if not a or not b:
            return 0.0
        embs = self._embedder.encode([a, b], show_progress_bar=False, normalize_embeddings=True)
        sim = float(self.cosine_similarity([embs[0]], [embs[1]])[0][0])  # [-1,1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def goals(self, mentee_goal: str, mentor_goal: str) -> float:
        if self.mode != "embed":
            return 0.0
        return self._embed_cosine01(mentee_goal, mentor_goal)

    def interests(self, mentee_interests: str, mentor_interests: str) -> float:
        if self.mode != "embed":
            return 0.0
        return self._embed_cosine01(mentee_interests, mentor_interests)

# Helpers
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

    # Same school
    same_school = str(mentor.get("School","")).strip().lower() == str(mentee.get("School","")).strip().lower()
    sschool = weights.get("same_school", 0) if same_school else 0
    score += sschool
    brk['same_school'] = sschool

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

    # Exact hobby overlap
    h_mentor = parse_list_cell(mentor.get("Interest/Hobbies"))
    h_mentee = parse_list_cell(mentee.get("Interest/Hobbies"))
    hobby_pts = len(h_mentor & h_mentee) * weights.get("hobby_overlap", 0)
    score += hobby_pts
    brk['hobby_overlap_exact'] = hobby_pts

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

# Matching algorithms
def greedy_assign(scores_df: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    used_mentors = set()
    used_mentees = set()
    for _, row in scores_df.sort_values('Score', ascending=False).iterrows():
        if row['Mentor'] in used_mentors or row['Mentee'] in used_mentees:
            continue
        assigned.append(row)
        used_mentors.add(row['Mentor'])
        used_mentees.add(row['Mentee'])
    if assigned:
        return pd.DataFrame(assigned).reset_index(drop=True)
    return pd.DataFrame(columns=scores_df.columns)

# Gale-Shapley stable matching as alternate
def gale_shapley(scores_df: pd.DataFrame) -> pd.DataFrame:
    mentee_groups = scores_df.groupby('Mentee')
    mentor_groups = scores_df.groupby('Mentor')

    mentee_prefs = {m: list(g.sort_values('Score', ascending=False)['Mentor']) for m, g in mentee_groups}
    mentor_prefs = {m: list(g.sort_values('Score', ascending=False)['Mentee']) for m, g in mentor_groups}
    mentor_rank = {m: {mentee: r for r, mentee in enumerate(prefs)} for m, prefs in mentor_prefs.items()}

    free_mentees = list(mentee_prefs.keys())
    next_idx = {m: 0 for m in free_mentees}
    match = {}

    while free_mentees:
        mtee = free_mentees.pop(0)
        prefs = mentee_prefs.get(mtee, [])
        if next_idx[mtee] >= len(prefs):
            continue
        mtor = prefs[next_idx[mtee]]
        next_idx[mtee] += 1

        if mtor not in match:
            match[mtor] = mtee
        else:
            other = match[mtor]
            if mentor_rank.get(mtor, {}).get(mtee, math.inf) < mentor_rank.get(mtor, {}).get(other, math.inf):
                match[mtor] = mtee
                free_mentees.append(other)
            else:
                free_mentees.append(mtee)

    pairs = []
    for mentor, mentee in match.items():
        rows = scores_df[(scores_df['Mentor'] == mentor) & (scores_df['Mentee'] == mentee)]
        if not rows.empty:
            pairs.append(rows.sort_values('Score', ascending=False).iloc[0])
    if pairs:
        return pd.DataFrame(pairs).reset_index(drop=True)
    return pd.DataFrame(columns=scores_df.columns)

# Simple sample data generator kept for testing
def generate_sample_data(n: int = 30) -> pd.DataFrame:
    import random
    genders = ["Male","Female","Nonbinary","Other"]
    ethnicities = ["Asian","Black","Hispanic","White","Mixed","Other"]
    mentor_status = ["Mentor","Mentee"]
    schools = ["Rutgers University","MIT","Stanford","Harvard","Princeton","Columbia"]
    mbti_types = list(MBTI_MATRIX.keys())
    hobbies = ["Coding","Music","Basketball","Reading","Gaming","Traveling","Volunteering"]
    goals_list = ["Get into tech","Improve leadership","Network more","Publish research","Prepare for grad school","Start a business"]
    firsts = ["Alex","Sam","Taylor","Jordan","Morgan","Chris","Jamie","Riley"]
    lasts  = ["Patel","Smith","Chen","Garcia","Khan","Williams","Brown","Lee"]

    rows = []
    for _ in range(n):
        first = random.choice(firsts)
        last = random.choice(lasts)
        name = f"{first} {last}"
        class_year = random.choice([2025,2026,2027,2028])
        email = f"{first.lower()}.{last.lower()}@{random.choice(['rutgers.edu','mit.edu','gmail.com'])}"
        gender = random.choice(genders)
        ethnicity = random.choice(ethnicities)
        role = random.choices(mentor_status, weights=[0.4,0.6])[0]
        school = random.choice(schools)
        mbti = random.choice(mbti_types)
        interests = ", ".join(random.sample(hobbies, k=random.randint(1,3)))
        goals = random.choice(goals_list)
        pref_gender = random.choice(genders+['No preference'])
        pref_eth = random.choice(ethnicities+['No preference'])
        state = random.choice(['NJ','NY','PA'])
        rows.append([name, class_year, email, gender, ethnicity, role, state, school, mbti, interests, goals, pref_gender, pref_eth])
    return pd.DataFrame(rows, columns=["Name","Class Year","Email","Gender","Ethnicity","Mentee/Mentor","state","School","MBTI","Interest/Hobbies","Goals","Preferred Gender","Preferred Ethnicity"])

# Optional Google Sheets import helper
def read_google_sheet(sheet_id: str, creds_json: str, worksheet_name: Optional[str] = None) -> pd.DataFrame:
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(creds_json, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1 if worksheet_name is None else sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)

# Write analysis tables
def write_analysis_tables(pairs_df: pd.DataFrame, df: pd.DataFrame, out_dir: Path):
    # Ensure output directory exists
    results_dir = out_dir / "results"
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
            "BRK::location_state", "BRK::same_school", "BRK::year_proximity", "BRK::hobby_overlap_exact",
            "BRK::mbti", "BRK::goals_rule", "BRK::goals_semantic", "BRK::interests_semantic"
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
    INPUT_CSV = 'sample_mentors_mentees.csv'  # e.g., 'responses.csv' or None to use sample data
    SHEET_ID = None   # e.g., 'your_google_sheet_id' or None to skip Google Sheets
    GCRED_PATH = None # e.g., 'service_account.json' if using Google Sheets
    OUTDIR = 'out'
    SEMANTIC = 'embed'  # 'off' or 'embed'
    EMBED_MODEL = 'all-MiniLM-L6-v2'
    PREFS = 'soft'    # 'soft' or 'hard';  how strictly to enforce gender and ethnicity preferences
    MATCH_ALGO = 'greedy'  # 'greedy' or 'stable'
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
        print(f"[INFO] Loaded {len(df)} rows from Google Sheet {SHEET_ID}")
    elif INPUT_CSV:
        df = pd.read_csv(INPUT_CSV)
        print(f"[INFO] Loaded CSV {INPUT_CSV} ({len(df)} rows)")
    else:
        df = generate_sample_data(40)
        print(f"[INFO] Using generated sample dataset ({len(df)} rows)")

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