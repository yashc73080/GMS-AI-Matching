import math
import pandas as pd

def greedy_assign(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns mentors to mentees based on the highest scores in a greedy manner.
    This function takes a DataFrame containing mentor-mentee pair scores and 
    assigns each mentor to a mentee such that no mentor or mentee is assigned 
    more than once. The assignment is performed by iterating through the 
    scores in descending order and selecting the highest available pair.
    """
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
    """
    Implements the Gale-Shapley algorithm to find stable matches between mentees and mentors
    based on their preference scores. The algorithm ensures that the resulting matches are stable, 
    meaning no mentee-mentor pair would prefer each other over their current matches.
    """
    
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

def greedy_assign_with_capacity(scores_df: pd.DataFrame, mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy assignment of mentees to mentors with capacity constraints.
    - Mentors take up to 'Num Mentees'.
    - Mentees can only be matched once.
    - Mentees have Num Mentees = NaN, ignore those values.
    """
    # Extract mentor capacities
    mentor_capacity = {}
    for _, row in mentors_df.iterrows():
        try:
            cap = int(row.get("Num Mentees"))
        except Exception:
            continue  # Skip if mentee (NaN or not int)
        if pd.isna(cap):
            continue
        mentor_capacity[row["Name"]] = cap

    mentor_assigned_count = {m: 0 for m in mentor_capacity}
    used_mentees = set()
    assigned = []

    # Assign greedily by descending score
    for _, row in scores_df.sort_values("Score", ascending=False).iterrows():
        mentor = row["Mentor"]
        mentee = row["Mentee"]

        if mentee in used_mentees:
            continue
        if mentor not in mentor_capacity:
            continue
        if mentor_assigned_count[mentor] < mentor_capacity[mentor]:
            assigned.append(row)
            mentor_assigned_count[mentor] += 1
            used_mentees.add(mentee)

    return pd.DataFrame(assigned).reset_index(drop=True) if assigned else pd.DataFrame(columns=scores_df.columns)

def gale_shapley_with_capacity(scores_df: pd.DataFrame, mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gale-Shapley hospital/residents algorithm with mentor capacity.
    - Mentors have capacity 'Num Mentees'.
    - Mentees are matched to at most one mentor.
    """
    # Mentor capacities
    mentor_capacity = {}
    for _, row in mentors_df.iterrows():
        try:
            cap = int(row.get("Num Mentees"))
        except Exception:
            continue
        if pd.isna(cap):
            continue
        mentor_capacity[row["Name"]] = cap

    # Build preference lists
    mentee_groups = scores_df.groupby("Mentee")
    mentor_groups = scores_df.groupby("Mentor")

    mentee_prefs = {m: list(g.sort_values("Score", ascending=False)["Mentor"]) for m, g in mentee_groups}
    mentor_prefs = {m: list(g.sort_values("Score", ascending=False)["Mentee"]) for m, g in mentor_groups}
    mentor_rank = {m: {mentee: r for r, mentee in enumerate(prefs)} for m, prefs in mentor_prefs.items()}

    # Track current matches
    matches = {m: [] for m in mentor_capacity}
    free_mentees = list(mentee_prefs.keys())
    next_idx = {m: 0 for m in free_mentees}

    while free_mentees:
        mtee = free_mentees.pop(0)
        prefs = mentee_prefs.get(mtee, [])
        if next_idx[mtee] >= len(prefs):
            continue
        mtor = prefs[next_idx[mtee]]
        next_idx[mtee] += 1

        if mtor not in mentor_capacity:
            continue

        if len(matches[mtor]) < mentor_capacity[mtor]:
            matches[mtor].append(mtee)
        else:
            # Mentor is full → check if prefers this mentee to its worst current match
            worst = max(matches[mtor], key=lambda x: mentor_rank.get(mtor, {}).get(x, float("inf")))
            if mentor_rank.get(mtor, {}).get(mtee, float("inf")) < mentor_rank.get(mtor, {}).get(worst, float("inf")):
                matches[mtor].remove(worst)
                matches[mtor].append(mtee)
                free_mentees.append(worst)
            else:
                free_mentees.append(mtee)

    # Convert matches to DataFrame rows
    pairs = []
    for mentor, mentees in matches.items():
        for mentee in mentees:
            rows = scores_df[(scores_df["Mentor"] == mentor) & (scores_df["Mentee"] == mentee)]
            if not rows.empty:
                pairs.append(rows.sort_values("Score", ascending=False).iloc[0])

    return pd.DataFrame(pairs).reset_index(drop=True) if pairs else pd.DataFrame(columns=scores_df.columns)
