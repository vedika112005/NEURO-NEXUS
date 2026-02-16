import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx

# ==========================================
# ⚙️ CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="NEURO-NEXUS | Neural Trace Dynamics",
    page_icon="🛰️",
    layout="wide"
)

# Load API Key from API.txt if it exists
def get_default_api_key():
    # 1. Try Streamlit Secrets (for Production/Cloud Deployment)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    # 2. Try Local File (for Local Development)
    try:
        with open("API.txt", "r") as f:
            return f.read().strip()
    except:
        return ""

DEFAULT_API_KEY = get_default_api_key()

# Custom CSS for Premium SOC Look
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E0E0E0; }
    
    /* Custom Card Style */
    .soc-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .critical-row { 
        background-color: #2b1111; 
        color: #ff4b4b; 
        border-left: 5px solid #ff4b4b; 
        padding: 12px; 
        margin-bottom: 8px; 
        border-radius: 4px;
        font-weight: 500;
    }
    
    .normal-row { 
        background-color: #0f1f18; 
        color: #00fa9a; 
        border-left: 5px solid #00fa9a; 
        padding: 12px; 
        margin-bottom: 8px; 
        border-radius: 4px;
    }
    
    .blocked-row {
        background-color: #1c1c1c;
        color: #666;
        text-decoration: line-through;
        border-left: 5px solid #444;
        padding: 12px;
        margin-bottom: 8px;
    }
    
    /* Header Enhancement */
    .main-header {
        font-family: 'Courier New', Courier, monospace;
        color: #58A6FF;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 2px solid #58A6FF;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛰️ AI & RAG LOGIC
# ==========================================

@st.cache_resource
def load_rag_engine():
    """Build or Load the FAISS Vector Database for historical context."""
    vector_db_path = "faiss_index"
    input_file = "enron_clean.csv"
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(vector_db_path):
        return FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        documents = df['topic'].dropna().head(1000).tolist()
        vector_db = FAISS.from_texts(documents, embeddings)
        vector_db.save_local(vector_db_path)
        return vector_db
    
    return None

def get_historical_context(query, vector_db):
    if not vector_db: return "No historical data available."
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([f"- {d.page_content}" for d in docs])

def analyze_threat_with_rag(row, api_key, vector_db):
    if not api_key: return "⚠️ API Key Missing"
    
    context = get_historical_context(row['subject'], vector_db)
    
    client = Groq(api_key=api_key)
    prompt = f"""
    You are a Senior Cyber Security Analyst. 
    Analyze this network packet using current metadata and historical context.
    
    CURRENT METADATA:
    - Sender: {row['sender']}
    - Subject: {row['subject']}
    - Alert Reason: {row['flag_reason']}
    
    HISTORICAL CONTEXT (Similar patterns found in legacy archives):
    {context}
    
    TASK:
    1. Assess the risk (e.g., Data Exfiltration, Spear Phishing, Insider Threat).
    2. Compare current activity with historical patterns.
    3. Severity level (Low/Med/High/Critical).
    4. Recommended SOC Response.
    """
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

def generate_smart_bait(row, api_key, vector_db):
    if not api_key: return "⚠️ API Key Missing"
    
    context = get_historical_context(row['subject'], vector_db)
    
    client = Groq(api_key=api_key)
    prompt = f"""
    You are a Counter-Intelligence Officer. 
    A user ({row['sender']}) is attempting a suspicious action: "{row['subject']}".
    
    HISTORICAL CONTEXT (Use this to make the bait sound authentic):
    {context}
    
    Write a deceptive "Bait Reply" that uses the terminology or style found in the historical context to keep the attacker engaged.
    The goal is to delay them and gain time for tracing. Keep it professional but subtly leading them on.
    """
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# ==========================================
# 🕸️ NETWORK VISUALIZATION LOGIC
# ==========================================
def generate_live_network(df):
    """Generates a force-directed graph of the last few interactions."""
    if df.empty: return None
    
    # Take last 20 interactions to keep it clean and fast
    recent_df = df.tail(20)
    
    # Create NetworkX graph: Sender -> Subject (Information Flow)
    G = nx.DiGraph()
    for _, row in recent_df.iterrows():
        G.add_edge(row['sender'], row['subject'], weight=1)
    
    nt = Network(
        height="400px", 
        width="100%", 
        bgcolor="#0B0E14", 
        font_color="#E0E0E0",
        directed=True
    )
    
    # Load from NetworkX
    nt.from_nx(G)
    
    # Customize look
    for node in nt.nodes:
        if "@" in node['id']: # It's a user
           node['color'] = '#58A6FF'
           node['shape'] = 'dot'
           node['size'] = 20
        else: # It's a subject/information packet
           node['color'] = '#FF4B4B' if any(res in node['id'] for res in ["Project", "Urgent", "Salaries", "Secret"]) else '#00FA9A'
           node['shape'] = 'diamond'
           node['size'] = 15
    
    # Force directed physics for the "moving" effect
    nt.toggle_physics(True)
    
    # Save and return path
    path = "tmp_network.html"
    nt.save_graph(path)
    return path

# ==========================================
# 🔄 DATA LOADING
# ==========================================

def load_live_data():
    """Robustly load the live stream CSV, handling potential race conditions with the server."""
    try: 
        # Skip bad lines to avoid crash on partial writes
        df = pd.read_csv("live_stream_data.csv", on_bad_lines='skip')
        # Ensure correct column names even if skipped lines were headers
        if list(df.columns) != ['timestamp', 'sender', 'subject', 'status', 'flag_reason']:
            df.columns = ['timestamp', 'sender', 'subject', 'status', 'flag_reason']
        return df
    except: 
        return pd.DataFrame(columns=['timestamp', 'sender', 'subject', 'status', 'flag_reason'])

def load_influence_scores():
    try: 
        df = pd.read_csv("user_influence_scores.csv")
        return df.sort_values(by="Influence_Score", ascending=False).head(10)
    except: return pd.DataFrame(columns=['User', 'Influence_Score'])

def load_risk_trends():
    try: 
        df = pd.read_csv("weekly_risk_trends.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except: return None

# ==========================================
# 🎮 STATE & SIDEBAR
# ==========================================

if 'show_blueprint' not in st.session_state:
    st.session_state['show_blueprint'] = False

if 'blocked_users' not in st.session_state:
    st.session_state['blocked_users'] = []

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00ADB5;'>🛰️ NEURO-NEXUS</h1>", unsafe_allow_html=True)
    st.caption("Neural Trace Dynamics")
    
    st.markdown("---")
    # 📘 PROJECT BLUEPRINT TOGGLE
    if st.button("📘 PROJECT BLUEPRINT", use_container_width=True):
        st.session_state['show_blueprint'] = not st.session_state['show_blueprint']
    
    if st.session_state['show_blueprint']:
        with st.expander("ℹ️ ABOUT NEURO-NEXUS", expanded=True):
            st.markdown("""
            **NEURO-NEXUS** is a state-of-the-art **Explainable Information Flow Tracking Dashboard**.
            
            ### 🛰️ The Mission
            The goal is simple: **Maximum Visibility.** We track the "Neural pathways" of information across your organization to identify anomalies before they become disasters.
            
            ### 🕸️ The Pillars of Intelligence
            1. **Macro Dynamics**: Real-time packet throughput and velocity analysis.
            2. **Micro Nexus**: Force-directed graphs showing hidden relationships between users and topics.
            3. **Forensic AI**: Using RAG (Retrieval-Augmented Generation) to give the AI a long-term "memory" of historical threats.
            
            ### 🛠️ Technology Stack
            - **Intelligence Engine**: Groq Llama-3 (70B)
            - **Pathways**: NetworkX & PyVis
            - **Memory**: FAISS Vector DB
            """)

    # 🛡️ QUARANTINE MANAGER
    with st.expander("🛡️ QUARANTINE MANAGER", expanded=False):
        st.markdown("### 🚫 Containment Protocol")
        st.caption("Quarantine instantly terminates a node's network connection. The system visually blocks and flags all traffic from these users until they are cleared.")
        
        st.markdown("---")
        if not st.session_state['blocked_users']:
            st.write("✅ *No users currently quarantined.*")
        else:
            for user in st.session_state['blocked_users']:
                uc1, uc2 = st.columns([4, 1])
                uc1.markdown(f"<span style='color: #FF4B4B; font-size: 0.9em; font-family: monospace;'>{user}</span>", unsafe_allow_html=True)
                if uc2.button("🔓", key=f"unblock_{user}", help=f"Restore connection for {user}"):
                    st.session_state['blocked_users'].remove(user)
                    st.toast(f"USER {user} CONNECTION RESTORED", icon="✅")
                    st.rerun()
    
    st.markdown("---")
    st.subheader("📶 System Controls")
    api_key = st.text_input("Groq API Key", type="password", value=DEFAULT_API_KEY if DEFAULT_API_KEY else "")
    is_paused = st.checkbox("⏸️ PAUSE FOR INVESTIGATION", value=False)
    
    if st.button("🗑️ Clear Blocklist", use_container_width=True):
        st.session_state['blocked_users'] = []
        st.rerun()

# Load Data
df = load_live_data()
influence_df = load_influence_scores()
vector_db = load_rag_engine()

# ==========================================
# 🖥️ MAIN DASHBOARD UI (TABBED NAVIGATION)
# ==========================================

st.markdown('<h1 class="main-header">🛰️ NEURO-NEXUS | Neural Trace Dynamics</h1>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_console, tab_dynamics = st.tabs(["🛰️ MAIN CONSOLE", "📈 NETWORK DYNAMICS & INTELLIGENCE"])

with tab_console:
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    crit_count = len(df[df['status'] == 'CRITICAL'])
    m1.metric("📡 Packets Analysed", len(df))
    m2.metric("🚨 Critical Threats", crit_count, delta=f"{crit_count} active", delta_color="inverse")
    m3.metric("🚫 Quarantined", len(st.session_state['blocked_users']))
    m4.metric("⚙️ RAG Engine", "Online" if vector_db else "Offline")

    st.markdown("---")

    # Split View: Feed and Intel
    col_feed, col_ops = st.columns([1, 1])

    with col_feed:
        st.subheader("🛰️ Traffic Metadata Feed")
        if not df.empty:
            display_df = df.tail(12).iloc[::-1]
            for _, row in display_df.iterrows():
                if row['sender'] in st.session_state['blocked_users']:
                    style, icon, detail = "blocked-row", "🚫", "CONNECTION TERMINATED"
                elif row['status'] == 'CRITICAL':
                    style, icon, detail = "critical-row", "🔥", f"ALERT: {row['flag_reason']}"
                else:
                    style, icon, detail = "normal-row", "📨", "Status: Verified Safe"
                
                st.markdown(f"""
                <div class="{style}">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75em; opacity: 0.7;">
                        <span>{row['timestamp']} | {row['sender']}</span>
                    </div>
                    <div style="font-size: 1.0em; margin: 3px 0;">{icon} <b>{row['subject']}</b></div>
                    <div style="font-size: 0.8em; opacity: 0.8;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for Live Server signal...")

    with col_ops:
        # Operations Tabs
        tab_inspect, tab_intel = st.tabs(["⚡ SOAR Operations", "🕵️ Network Intelligence"])
        
        with tab_inspect:
            if not is_paused:
                st.warning("⚠️ PAUSE STREAM to access SOAR controls.")
                st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJndXIzZ3R6Z3R6Z3R6Z3R6Z3R6Z3R6Z3R6Z3R6Z3R6Z3R6JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/L1R1TVTh2RhtR5Jgww/giphy.gif", width=150)
            else:
                threats = df[df['status'] == 'CRITICAL'].tail(20).iloc[::-1]
                if threats.empty:
                    st.success("No active threats to remediate.")
                else:
                    target_options = threats.apply(lambda x: f"{x['sender']} | {x['subject']}", axis=1)
                    selected_target = st.selectbox("Select Threat for Action:", target_options)
                    
                    idx = target_options[target_options == selected_target].index[0]
                    target_row = df.loc[idx]
                    
                    user_score = influence_df[influence_df['User'] == target_row['sender']]
                    if not user_score.empty:
                        st.error(f"⚠️ **HIGH VALUE TARGET:** This user has an Influence Score of {user_score['Influence_Score'].values[0]:.1f}")

                    b_col1, b_col2 = st.columns(2)
                    if b_col1.button("🔍 Run Forensic AI", use_container_width=True, type="primary"):
                        with st.spinner("🤖 Investigating History..."):
                            report = analyze_threat_with_rag(target_row, api_key, vector_db)
                            st.markdown(f'<div class="soc-card"><h4 style="color: #58A6FF;">Forensic Report</h4><hr style="border-color: #30363D;">{report}</div>', unsafe_allow_html=True)
                    
                    if b_col2.button("🎣 Generate Neural Bait", use_container_width=True):
                        with st.spinner("🛰️ Synthesizing Bait..."):
                            bait = generate_smart_bait(target_row, api_key, vector_db)
                            st.code(bait)

                    st.markdown("---")
                    if st.button("🚫 QUARANTINE USER", use_container_width=True, type="secondary"):
                        if target_row['sender'] not in st.session_state['blocked_users']:
                            st.session_state['blocked_users'].append(target_row['sender'])
                            st.toast(f"USER {target_row['sender']} QUARANTINED", icon="🚫")
                            st.rerun()

        with tab_intel:
            st.subheader("💎 Highest Risk Influencers")
            st.write("Users with the most connections. If compromised, they represent the highest organizational risk.")
            if not influence_df.empty:
                st.dataframe(influence_df, hide_index=True, use_container_width=True)

with tab_dynamics:
    st.subheader("📊 Full-Spectrum Network Intelligence")
    
    col_graph, col_pulse = st.columns([1.5, 1])

    with col_graph:
        st.markdown("### 🕸️ Live Force-Directed Graph")
        st.caption("Dots = Users | Diamonds = Information Packets (Red is Risky)")
        
        html_path = generate_live_network(df)
        if html_path:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=450)
        else:
            st.info("Gathering node data for visualization...")

    with col_pulse:
        st.markdown("### 📡 Live Signal Pulse")
        if not df.empty:
            # Robust conversion: handle potential race conditions/corrupted strings from live file
            df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S', errors='coerce')
            df = df.dropna(subset=['dt']) # Remove rows that failed conversion
            
            pulse_data = df.groupby('timestamp').size().reset_index(name='packets')
            pulse_window = pulse_data.tail(30)
            
            # High-Fidelity Signal Graph
            fig_large = px.line(
                pulse_window, 
                x='timestamp', 
                y='packets',
                markers=True,
                color_discrete_sequence=['#00ADB5']
            )
            fig_large.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E0E0E0",
                height=300,
                xaxis=dict(showgrid=False, title="Simulation Time"),
                yaxis=dict(showgrid=True, gridcolor="#30363D", title="Packet Velocity"),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            fig_large.update_traces(line_width=3, line_color="#00ADB5", marker=dict(size=8, color="#58A6FF", symbol="diamond"))
            st.plotly_chart(fig_large, use_container_width=True)

            # plain English Analysis
            st.markdown('<div class="soc-card">', unsafe_allow_html=True)
            avg_vol = pulse_window['packets'].mean()
            st.write(f"**Mean Traffic Flow:** `{avg_vol:.2f} pkts/sec`")
            
            if crit_count > 5:
                st.error("🚨 **CRITICAL ANOMALY:** High density of suspicious nodes found in graph.")
            else:
                st.success("✅ **STABLE SYNC:** Network topology is within safe boundaries.")
            st.markdown('</div>', unsafe_allow_html=True)
            
# ==========================================
# 🔄 REFRESH ENGINE
# ==========================================
if not is_paused:
    time.sleep(1.5)
    st.rerun()
