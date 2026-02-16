import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import networkx as nx

print("🚀 REBOOTING INTELLIGENCE PIPELINE (Realistic Model)...")

# ==========================================
# 1. GENERATE REALISTIC DATA (Departmental Model)
# ==========================================
print("Step 1: Simulating corporate structure...")

# Create 20 users with roles
# VIPs (Connected to everyone)
vips = ["ceo@company.com", "cto@company.com"]
# Department Leads (Connected within dept and to VIPs)
dept_leads = ["hr_lead@company.com", "eng_lead@company.com", "sales_lead@company.com"]
# Regular Staff
staff = [f"staff_{i:02d}@company.com" for i in range(1, 16)]

users = vips + dept_leads + staff
depts = {
    "HR": ["hr_lead@company.com", "staff_01@company.com", "staff_02@company.com", "staff_03@company.com"],
    "ENG": ["eng_lead@company.com", "staff_04@company.com", "staff_05@company.com", "staff_06@company.com", "staff_07@company.com"],
    "SALES": ["sales_lead@company.com", "staff_08@company.com", "staff_09@company.com", "staff_10@company.com", "staff_11@company.com"]
}

emails = []
start_date = datetime(2024, 1, 1)

# Generate 5000 emails to ensure strong signals
for _ in range(5000):
    date = start_date + timedelta(days=random.randint(0, 60))
    
    # Logic for interaction:
    coin = random.random()
    if coin < 0.2: # 20% involve VIPs (Central Hubs)
        sender = random.choice(vips)
        receiver = random.choice(users)
    elif coin < 0.5: # 30% intra-departmental
        dept = random.choice(list(depts.keys()))
        sender = random.choice(depts[dept])
        receiver = random.choice(depts[dept])
    else: # 50% General company-wide noise
        sender = random.choice(users)
        receiver = random.choice(users)

    if sender == receiver: continue
    
    topic = random.choice(["Sync", "Update", "Report", "Planning", "Coffee"])
    
    # Inject anomaly for STAFF_05 (Insider Threat)
    if sender == "staff_05@company.com" and random.random() > 0.8:
        topic = "CONFIDENTIAL: Internal Data Export"
        
    emails.append([sender, receiver, date, topic])

df_flow = pd.DataFrame(emails, columns=['sender', 'receiver', 'timestamp', 'topic'])
df_flow.to_csv("enron_clean.csv", index=False)
print(f"   ✅ {len(df_flow)} interactions generated.")

# ==========================================
# 2. RUN NETWORK ANALYSIS (Centrality)
# ==========================================
print("Step 2: Calculating Network Intelligence...")
G = nx.from_pandas_edgelist(df_flow, 'sender', 'receiver', create_using=nx.DiGraph())

# Activity = Normalized Degree
degree = nx.degree_centrality(G)
# Influence = Betweenness (Bridges)
betweenness = nx.betweenness_centrality(G)

df_scores = pd.DataFrame({
    'User': list(degree.keys()),
    'Activity_Score': [degree[u]*100 for u in degree.keys()],
    'Influence_Score': [betweenness[u]*100 for u in degree.keys()]
}).sort_values('Influence_Score', ascending=False)

df_scores.to_csv("user_influence_scores.csv", index=False)
print("   ✅ Scores calculated (VIPs and Dept Leads should now be top).")

# ==========================================
# 3. RUN TEMPORAL RISK ANALYSIS
# ==========================================
print("Step 3: Analyzing Temporal Spikes...")
df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'])
df_trends = df_flow.groupby([pd.Grouper(key='timestamp', freq='D'), 'sender']).size().reset_index(name='msg_count')

def detect_spike(df):
    mean = df['msg_count'].mean()
    std = df['msg_count'].std()
    if pd.isna(std) or std == 0: std = 1
    df['anomaly_threshold'] = mean + (2.0 * std) 
    df['risk_level'] = np.where(df['msg_count'] > df['anomaly_threshold'], 'High', 'Normal')
    return df

df_trends = df_trends.groupby('sender', group_keys=False).apply(detect_spike)
df_trends.to_csv("weekly_risk_trends.csv", index=False)

risks = len(df_trends[df_trends['risk_level'] == 'High'])
print(f"   ✅ Risk analysis done. FOUND {risks} ANOMALIES.")
print("\n🚀 PIPELINE REBOOTED. VIEW DASHBOARD FOR UPDATED INTEL.")
