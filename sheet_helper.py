from typing import Optional
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def normalize_responses(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Full Name": "Name",
        "Email (School or University Email)": "Email",
        "Do you want to be a Mentor or Mentee?": "Mentee/Mentor",
        "How many mentees do you want?": "Num Mentees",
    }
    df = df.rename(columns=rename_map)

    # Treat empty strings as NaN for merging
    for col in ["University", "High School"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)

    # Merge into one School column, preferring University, then High School
    if "University" in df.columns and "High School" in df.columns:
        df["School"] = df["University"].combine_first(df["High School"])
        df = df.drop(columns=["University", "High School"], errors="ignore")
    elif "University" in df.columns:
        df["School"] = df["University"]
        df = df.drop(columns=["University"], errors="ignore")
    elif "High School" in df.columns:
        df["School"] = df["High School"]
        df = df.drop(columns=["High School"], errors="ignore")

    return df

# Google Sheets import helper
def read_google_sheet(sheet_id: str, creds_json: str, worksheet_name: Optional[str] = None) -> pd.DataFrame:
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(creds_json, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1 if worksheet_name is None else sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)