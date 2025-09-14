from typing import Optional
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

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

# Google Sheets import helper
def read_google_sheet(sheet_id: str, creds_json: str, worksheet_name: Optional[str] = None) -> pd.DataFrame:
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(creds_json, scopes=scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1 if worksheet_name is None else sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)