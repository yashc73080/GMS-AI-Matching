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