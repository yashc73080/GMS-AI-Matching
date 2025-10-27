import os
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()

SHEET_ID = os.getenv('SHEET_ID')
GCRED_PATH = os.getenv('GCRED_PATH')

if __name__ == "__main__":
    print("Testing Google Sheets connection...")
    print(f"Sheet ID: {SHEET_ID}")
    print(f"Credentials: {GCRED_PATH}")
    print("-" * 60)
    
    try:
        # Connect to Google Sheets
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(GCRED_PATH, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(SHEET_ID)
        
        print(f"\n✅ Successfully connected to Google Sheet!")
        print(f"Spreadsheet title: {sh.title}")
        print(f"\nAvailable worksheets:")
        
        for i, worksheet in enumerate(sh.worksheets(), 1):
            print(f"  {i}. {worksheet.title} (id: {worksheet.id}, {worksheet.row_count} rows x {worksheet.col_count} cols)")
        
        # Try to read from the first worksheet with data
        print("\n" + "=" * 60)
        for worksheet in sh.worksheets():
            print(f"\nReading worksheet: '{worksheet.title}'")
            print("-" * 60)
            
            data = worksheet.get_all_records()
            
            if not data:
                print(f"  ⚠️  No data found (empty or only headers)")
                # Try to get all values including headers
                all_values = worksheet.get_all_values()
                if all_values:
                    print(f"  Headers found: {all_values[0]}")
                continue
            
            print(f"  ✅ Found {len(data)} rows of data")
            print(f"  Columns: {list(data[0].keys())}")
            print(f"\n  First row:")
            for key, value in data[0].items():
                print(f"    {key}: {value}")
            break
        
    except Exception as e:
        print(f"\n❌ Error connecting to Google Sheet:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()