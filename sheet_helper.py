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
        "Are you part of the LGBTQ+ community?": "LGBTQ+",
        "Would you like to be matched with someone of similar ethnic demographics?": "Prefer Similar Ethnicity",
        "Would you like to be matched with someone of similar gender?": "Prefer Similar Gender",
        "Would you like to be matched with someone from the LGBTQ+ community?": "Prefer LGBTQ+",
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
    
    # If no worksheet name is provided, try to find the first worksheet with data
    if worksheet_name is None:
        # Try all worksheets and use the first one with data
        for ws in sh.worksheets():
            data = ws.get_all_records()
            if data:  # Found a worksheet with data
                print(f"[INFO] Reading from worksheet: '{ws.title}'")
                return pd.DataFrame(data)
        # If no worksheet has data, fall back to sheet1
        print(f"[WARN] No worksheets contain data. Using first sheet: '{sh.sheet1.title}'")
        ws = sh.sheet1
    else:
        ws = sh.worksheet(worksheet_name)
        print(f"[INFO] Reading from worksheet: '{worksheet_name}'")
    
    data = ws.get_all_records()
    return pd.DataFrame(data)