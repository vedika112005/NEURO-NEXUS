import pandas as pd
import time
import random
import os
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
LIVE_FILE = "live_stream_data.csv"

# 1. Normal Business Traffic
SAFE_SUBJECTS = [
    "Re: Weekly Sync", "Lunch plans?", "Invoice 4022 attached", 
    "Meeting notes", "Q3 Forecast updates", "Happy Birthday!", 
    "Jira Ticket #992", "Out of Office", "Coffee?", "Client feedback"
]

# 2. Threat Scenarios (The AI will catch these)
RISK_SCENARIOS = [
    {"subject": "CONFIDENTIAL: Project Falcon Data", "reason": "DLP Keyword Match"},
    {"subject": "Wire Transfer Request - Urgent", "reason": "Financial Fraud Pattern"},
    {"subject": "FW: Employee Salaries List", "reason": "PII Data Leakage"},
    {"subject": "Backup_database.zip", "reason": "Large File Exfiltration"},
    {"subject": "Secret merger details", "reason": "Insider Trading Watchlist"}
]

# Mock Users
USERS = ["user_01@company.com", "user_02@company.com", "user_05@company.com", "admin@company.com"]

print("LIVE SERVER: Initializing Traffic Stream...")

# Reset the file header
df_cols = pd.DataFrame(columns=['timestamp', 'sender', 'subject', 'status', 'flag_reason'])
df_cols.to_csv(LIVE_FILE, index=False)

print("SERVER LIVE. GENERATING TRAFFIC...")

try:
    while True:
        sender = random.choice(USERS)
        
        # 15% Chance of a Threat
        if random.random() < 0.15:
            scenario = random.choice(RISK_SCENARIOS)
            subject = scenario["subject"]
            status = "CRITICAL"
            reason = scenario["reason"]
        else:
            subject = random.choice(SAFE_SUBJECTS)
            status = "NORMAL"
            reason = "-"

        # Create the data row
        new_row = pd.DataFrame([{
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'sender': sender,
            'subject': subject,
            'status': status,
            'flag_reason': reason
        }])
        
        # Append to CSV
        new_row.to_csv(LIVE_FILE, mode='a', header=False, index=False)
        
        # Print to terminal so you know it's working
        icon = "[CRITICAL]" if status == "CRITICAL" else "[NORMAL]"
        print(f"{icon} {sender}: {subject}")
        
        # Wait 1-2 seconds between emails
        time.sleep(random.uniform(1.0, 2.0))

except KeyboardInterrupt:
    print("\nSERVER STOPPED.")