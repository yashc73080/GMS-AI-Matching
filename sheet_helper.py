import pandas as pd

def normalize_responses(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Full Name": "Name",
        "Email (School or University Email)": "Email",
        "Do you want to be a Mentor or Mentee?": "Mentor/Mentee",
        "How many mentees do you want?": "Num Mentees",
        "University": "School",
        "High School": "School"
    }
    df = df.rename(columns=rename_map)
    
    # If both University and High School exist, merge into one School column
    if "University" in df.columns and "High School" in df.columns:
        df["School"] = df["University"].fillna(df["High School"])
    elif "University" in df.columns:
        df["School"] = df["University"]
    elif "High School" in df.columns:
        df["School"] = df["High School"]
    
    return df
