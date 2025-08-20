#!/usr/bin/env python3
# mentor_mentee_matcher.py
# Matching mentors/mentees with a hardcoded 16x16 MBTI matrix + optional semantic scoring.
# If --input is omitted, a sample dataset (sample_df) is auto-generated.

import argparse
import math
import re
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd

# -----------------------
# Hardcoded MBTI matrix (0..15)
# -----------------------
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

# -----------------------
# Configurable weights
# -----------------------
DEFAULT_WEIGHTS = {
    "gender_pref": 30,
    "eth_pref": 30,
    "same_school": 20,
    "year_proximity": 20,
    "hobby_overlap": 10,
    "mbti": 15,              # matrix already 0..15
    "goals_rule": 8,         # small rules bonus
    "goals_semantic": 20,    # semantic bonus scale (0..20)
    "interests_semantic": 10 # semantic bonus scale (0..10)
}

# -----------------------
# Optional semantic scoring (off by default)
# -----------------------
class SemanticScorer:
    """
    mode:
      - 'off'   : no semantic scoring
      - 'embed' : sentence-transformers embeddings similarity (install deps)
      - 'llm'   : call your LLM API (you implement _llm_score)
    """
    def __init__(self, mode="off", model_name="all-MiniLM-L6-v2", llm_api: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.model_name = model_name
        self.llm_api = llm_api or {}
        self._embedder = None
        if self.mode == "embed":
            try:
                from sentence_transformers import SentenceTransformer
                self.SentenceTransformer = SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
                from sklearn.metrics.pairwise import cosine_similarity
                self.cosine_similarity = cosine_similarity
            except Exception:
                raise RuntimeError("Embedding mode requires: pip install sentence-transformers scikit-learn")

    @staticmethod
    def _norm(s): return "" if s is None else str(s).strip()

    def _embed_cosine01(self, a: str, b: str) -> float:
        a = self._norm(a); b = self._norm(b)
        if not a or not b:
            return 0.0
        embs = self._embedder.encode([a, b], show_progress_bar=False, normalize_embeddings=True)
        sim = float(self.cosine_similarity([embs[0]], [embs[1]])[0][0])  # [-1,1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def _llm_score(self, mentee_text: str, mentor_text: str, instruction: str) -> float:
        # TODO: replace with your LLM provider call; must return 0..1
        a = self._norm(mentee_text).lower()
        b = self._norm(mentor_text).lower()
        if not a or not b:
            return 0.0
        if a == b:
            return 0.9
        ta = set(re.findall(r"\w+", a))
        tb = set(re.findall(r"\w+", b))
        if not ta or not tb:
            return 0.0
        overlap = len(ta & tb) / min(len(ta), len(tb))
        return max(0.0, min(1.0, 0.3 + 0.7 * overlap))

    def goals(self, mentee_goal: str, mentor_goal: str) -> float:
        if self.mode == "off": return 0.0
        if self.mode == "embed": return self._embed_cosine01(mentee_goal, mentor_goal)
        if self.mode == "llm": return self._llm_score(mentee_goal, mentor_goal, "Score goals alignment 0..1")
        return 0.0

    def interests(self, mentee_interests: str, mentor_interests: str) -> float:
        if self.mode == "off": return 0.0
        if self.mode == "embed": return self._embed_cosine01(mentee_interests, mentor_interests)
        if self.mode == "llm": return self._llm_score(mentee_interests, mentor_interests, "Score interests relatedness 0..1")
        return 0.0

# -----------------------
# Helpers
# -----------------------
def parse_list_cell(cell):
    if pd.isna(cell):
        return set()
    items = re.split(r"[,/]", str(cell))
    return set([it.strip().title() for it in items if it.strip()])

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

# -----------------------
# Scoring
# -----------------------
def compute_score(mentor, mentee, sem: SemanticScorer, weights=None, prefs_mode="soft") -> Tuple[float, Dict[str, float]]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if str(mentor.get("Mentor/Mentee")) != "Mentor" or str(mentee.get("Mentor/Mentee")) != "Mentee":
        return -1.0, {}

    if prefs_mode == "hard":
        if not preference_satisfied(mentee.get("Preferred Gender"), mentor.get("Gender")): return -1.0, {}
        if not preference_satisfied(mentee.get("Preferred Ethnicity"), mentor.get("Ethnicity")): return -1.0, {}
        if not preference_satisfied(mentor.get("Preferred Gender"), mentee.get("Gender")): return -1.0, {}
        if not preference_satisfied(mentor.get("Preferred Ethnicity"), mentee.get("Ethnicity")): return -1.0, {}

    score = 0.0
    brk = {}

    gpref = weights["gender_pref"] if preference_satisfied(mentee.get("Preferred Gender"), mentor.get("Gender")) else 0
    epref = weights["eth_pref"] if preference_satisfied(mentee.get("Preferred Ethnicity"), mentor.get("Ethnicity")) else 0
    score += gpref + epref
    brk["pref_gender"] = gpref
    brk["pref_eth"] = epref

    mgpref = 0.5 * weights["gender_pref"] if preference_satisfied(mentor.get("Preferred Gender"), mentee.get("Gender")) else 0
    mepref = 0.5 * weights["eth_pref"]   if preference_satisfied(mentor.get("Preferred Ethnicity"), mentee.get("Ethnicity")) else 0
    score += mgpref + mepref
    brk["mentor_pref_gender"] = mgpref
    brk["mentor_pref_eth"] = mepref

    same_school = str(mentor.get("School","")).strip().lower() == str(mentee.get("School","")).strip().lower()
    sschool = weights["same_school"] if same_school else 0
    score += sschool
    brk["same_school"] = sschool

    yp = 0
    try:
        y1 = int(str(mentor.get("Class Year","")).strip())
        y2 = int(str(mentee.get("Class Year","")).strip())
        if abs(y1 - y2) <= 2:
            yp = weights["year_proximity"]
    except Exception:
        pass
    score += yp
    brk["year_proximity"] = yp

    h_mentor = parse_list_cell(mentor.get("Interest/Hobbies"))
    h_mentee = parse_list_cell(mentee.get("Interest/Hobbies"))
    hobby_pts = len(h_mentor & h_mentee) * weights["hobby_overlap"]
    score += hobby_pts
    brk["hobby_overlap_exact"] = hobby_pts

    mbti_pts = mbti_score_full(mentor.get("MBTI",""), mentee.get("MBTI",""))
    score += mbti_pts
    brk["mbti"] = mbti_pts

    gr = goals_alignment_rules(mentee.get("Goals"), mentor.get("Goals"))
    score += gr
    brk["goals_rule"] = gr

    gs01 = sem.goals(mentee.get("Goals"), mentor.get("Goals"))  # 0..1
    gs_pts = gs01 * weights["goals_semantic"]
    score += gs_pts
    brk["goals_semantic"] = gs_pts

    isem01 = sem.interests(mentee.get("Interest/Hobbies"), mentor.get("Interest/Hobbies"))  # 0..1
    isem_pts = isem01 * weights["interests_semantic"]
    score += isem_pts
    brk["interests_semantic"] = isem_pts

    return float(round(score, 2)), brk

# -----------------------
# Build score table
# -----------------------
def build_match_table(df, sem: SemanticScorer, weights=None, prefs_mode="soft"):
    mentors = df[df["Mentor/Mentee"] == "Mentor"].reset_index(drop=True)
    mentees = df[df["Mentor/Mentee"] == "Mentee"].reset_index(drop=True)

    rows = []
    for _, me in mentees.iterrows():
        for _, mr in mentors.iterrows():
            s, br = compute_score(mr, me, sem=sem, weights=weights, prefs_mode=prefs_mode)
            if s >= 0:
                rows.append({
                    "Mentee": me["Name"],
                    "Mentee Email": me["Email"],
                    "Mentor": mr["Name"],
                    "Mentor Email": mr["Email"],
                    "Score": s,
                    "Shared Hobbies (Exact)": ", ".join(sorted(parse_list_cell(mr.get("Interest/Hobbies")) & parse_list_cell(me.get("Interest/Hobbies")))),
                    "Same School": str(mr.get("School","")).strip().lower() == str(me.get("School","")).strip().lower(),
                    "Year Gap": (
                        abs(int(mr.get("Class Year")) - int(me.get("Class Year")))
                        if str(mr.get("Class Year")).isdigit() and str(me.get("Class Year")).isdigit() else None
                    ),
                    "MBTI Mentor/Mentee": f"{mr.get('MBTI','')} / {me.get('MBTI','')}",
                    "Goals (Mentor←Mentee)": f"{mr.get('Goals','')} ← {me.get('Goals','')}",
                    **{f"BRK::{k}": v for k, v in br.items()}
                })
    scores_df = pd.DataFrame(rows).sort_values(["Mentee", "Score"], ascending=[True, False]).reset_index(drop=True)
    return mentors, mentees, scores_df

# -----------------------
# Matching algorithms
# -----------------------
def greedy_assign(scores_df: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    used_mentors = set()
    used_mentees = set()
    for _, row in scores_df.sort_values("Score", ascending=False).iterrows():
        if row["Mentor"] in used_mentors or row["Mentee"] in used_mentees:
            continue
        assigned.append(row)
        used_mentors.add(row["Mentor"])
        used_mentees.add(row["Mentee"])
    return pd.DataFrame(assigned)

def gale_shapley(scores_df: pd.DataFrame) -> pd.DataFrame:
    mentee_groups = scores_df.groupby("Mentee")
    mentor_groups = scores_df.groupby("Mentor")

    mentee_prefs = {m: list(g.sort_values("Score", ascending=False)["Mentor"]) for m, g in mentee_groups}
    mentor_prefs = {m: list(g.sort_values("Score", ascending=False)["Mentee"]) for m, g in mentor_groups}
    mentor_rank = {m: {mentee: r for r, mentee in enumerate(prefs)} for m, prefs in mentor_prefs.items()}

    free_mentees = list(mentee_prefs.keys())
    next_idx = {m: 0 for m in free_mentees}
    match = {}  # mentor -> mentee

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
        rows = scores_df[(scores_df["Mentor"] == mentor) & (scores_df["Mentee"] == mentee)]
        if not rows.empty:
            pairs.append(rows.sort_values("Score", ascending=False).iloc[0])
    return pd.DataFrame(pairs)

# -----------------------
# Sample data generator (used when --input is omitted)
# -----------------------
def generate_sample_data(n: int = 30) -> pd.DataFrame:
    genders = ["Male", "Female", "Nonbinary", "Other"]
    ethnicities = ["Asian", "Black", "Hispanic", "White", "Mixed", "Other"]
    mentor_status = ["Mentor", "Mentee"]
    schools = ["Rutgers University", "MIT", "Stanford", "Harvard", "Princeton", "Columbia"]
    mbti_types = [
        "INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
        "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"
    ]
    hobbies = ["Coding", "Music", "Basketball", "Reading", "Gaming", "Traveling", "Volunteering"]
    goals_list = [
        "Get into tech", "Improve leadership", "Network more",
        "Publish research", "Prepare for grad school", "Start a business"
    ]
    firsts = ["Alex","Sam","Taylor","Jordan","Morgan","Chris","Jamie","Riley"]
    lasts  = ["Patel","Smith","Chen","Garcia","Khan","Williams","Brown","Lee"]

    rows: List[List] = []
    for _ in range(n):
        first = random.choice(firsts)
        last = random.choice(lasts)
        name = f"{first} {last}"
        age = random.randint(18, 30)
        class_year = random.choice([2025, 2026, 2027, 2028])
        email = f"{first.lower()}.{last.lower()}@{random.choice(['rutgers.edu','mit.edu','stanford.edu'])}"
        gender = random.choice(genders)
        ethnicity = random.choice(ethnicities)
        role = random.choices(mentor_status, weights=[0.4, 0.6])[0]
        school = random.choice(schools)
        mbti = random.choice(mbti_types)
        interests = ", ".join(random.sample(hobbies, k=random.randint(1,3)))
        goals = random.choice(goals_list)
        pref_gender = random.choice(genders)  # or "No preference"
        pref_eth = random.choice(ethnicities) # or "No preference"
        rows.append([name, age, class_year, email, gender, ethnicity, role,
                     school, mbti, interests, goals, pref_gender, pref_eth])

    return pd.DataFrame(rows, columns=[
        "Name","Age","Class Year","Email","Gender","Ethnicity","Mentor/Mentee",
        "School","MBTI","Interest/Hobbies","Goals","Preferred Gender","Preferred Ethnicity"
    ])

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Mentor–Mentee Matching (MBTI matrix + optional semantics). If --input omitted, uses a generated sample_df.")
    ap.add_argument("--input", "-i", help="Input CSV path (omit to auto-generate a sample dataset)")
    ap.add_argument("--outdir", "-o", default=".", help="Output directory")
    ap.add_argument("--prefs", choices=["soft","hard"], default="soft", help="Preference handling")
    ap.add_argument("--match", choices=["greedy","stable"], default="greedy", help="Matching algorithm")
    ap.add_argument("--topk", type=int, default=3, help="Top-K per mentee to export")
    ap.add_argument("--semantic", choices=["off","embed","llm"], default="off", help="Semantic scoring for Goals/Interests")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model (if --semantic=embed)")
    ap.add_argument("--llm-model", default="your-llm-name", help="LLM model name (if --semantic=llm)")
    ap.add_argument("--llm-endpoint", default="https://api.your-llm.com/chat", help="LLM endpoint (placeholder)")
    ap.add_argument("--rows", type=int, default=30, help="Rows to generate for sample_df when --input is omitted")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
        print(f"[INFO] Loaded input CSV: {args.input} ({len(df)} rows)")
    else:
        df = generate_sample_data(args.rows)
        print(f"[INFO] Using generated sample_df with {len(df)} rows (no --input provided)")

    expected = [
        "Name", "Email", "Mentor/Mentee", "Gender", "Ethnicity", "School",
        "Class Year", "MBTI", "Interest/Hobbies", "Goals",
        "Preferred Gender", "Preferred Ethnicity"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")

    if args.semantic == "embed":
        sem = SemanticScorer(mode="embed", model_name=args.embed_model)
    elif args.semantic == "llm":
        sem = SemanticScorer(mode="llm", llm_api={"endpoint": args.llm_endpoint, "model": args.llm_model})
    else:
        sem = SemanticScorer(mode="off")

    mentors, mentees, scores_df = build_match_table(df, sem=sem, weights=DEFAULT_WEIGHTS, prefs_mode=args.prefs)

    full_scores = out_dir / "mentor_mentee_full_scores.csv"
    scores_df.to_csv(full_scores, index=False)

    topk = scores_df.groupby("Mentee").head(args.topk).reset_index(drop=True)
    topk_path = out_dir / f"mentor_mentee_top{args.topk}_per_mentee.csv"
    topk.to_csv(topk_path, index=False)

    if args.match == "greedy":
        pairs = greedy_assign(scores_df)
    else:
        pairs = gale_shapley(scores_df)
    pairs_path = out_dir / f"mentor_mentee_assigned_pairs_{args.match}.csv"
    pairs.to_csv(pairs_path, index=False)

    print("[OK] Wrote:")
    print(f"  - {full_scores}")
    print(f"  - {topk_path}")
    print(f"  - {pairs_path}")
    print(f"[INFO] Mentors: {len(mentors)}, Mentees: {len(mentees)}, Scored pairs: {len(scores_df)}")
    if not pairs.empty:
        print("[INFO] Sample assigned pairs:")
        print(pairs[["Mentee","Mentor","Score"]].head().to_string(index=False))

if __name__ == "__main__":
    main()