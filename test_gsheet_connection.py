import os
from dotenv import load_dotenv
import gspread

# Load environment variables from .env
load_dotenv()

SHEET_ID = os.getenv('SHEET_ID')
GCRED_PATH = os.getenv('GCRED_PATH')

# Authenticate and connect to Google Sheets
gc = gspread.service_account(filename=GCRED_PATH)
sh = gc.open_by_key(SHEET_ID)

# Get the first worksheet
worksheet = sh.get_worksheet(0)

# Fetch all data
data = worksheet.get_all_values()

# Print all contents
for row in data:
    print(row)