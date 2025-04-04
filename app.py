import streamlit as st
import pandas as pd

st.set_page_config(page_title="FairRank", layout="wide")
st.title("FairRank: Merit-Based Hiring with Context")

# STEP 1: Load data from GitHub raw CSV
CSV_URL = "https://raw.githubusercontent.com/NG1897/Merit-Algorithm/main/student_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load CSV from GitHub: {e}")
    st.stop()

# STEP 2: Define scoring weights
GPA_WEIGHT = 0.50
PROJECT_WEIGHT = 0.25
LEADERSHIP_WEIGHT = 0.15
INTERNSHIP_BONUS = 0.05
MISDEMEANOR_PENALTY = 0.05  # Per severity point

def compute_merit(row):
    base_score = (
        row['GPA'] * GPA_WEIGHT +
        row['ProjectsScore'] * PROJECT_WEIGHT +
        row['LeadershipScore'] * LEADERSHIP_WEIGHT +
        (INTERNSHIP_BONUS if row['InternshipExp'] else 0)
    )

    # Contextual boosts
    econ_boost = 0.10 if row['EconomicStatus'] == 'Low' else 0.05 if row['EconomicStatus'] == 'Medium' else 0.0
    board_boost = 0.05 if row['SchoolBoard'] == 'StateBoard' else 0.0
    setback_boost = 0.15 if row['SetbackFlag'] else 0.0

    # Misdemeanor penalty
    penalty = MISDEMEANOR_PENALTY * row['MisdemeanorSeverity']

    final_score = base_score * (1 + econ_boost + board_boost + setback_boost) * (1 - penalty)
    return final_score

# STEP 3: Apply scoring and rank
df['MeritScore'] = df.apply(compute_merit, axis=1)
df['Rank'] = df['MeritScore'].rank(ascending=False, method='first').astype(int)
df = df.sort_values(by='MeritScore', ascending=False)

# STEP 4: Display results
st.subheader("Final Ranked Students")
st.dataframe(df.reset_index(drop=True), use_container_width=True)

# Optional: Highlight top 5
st.subheader("Top 5 Candidates")
st.table(df[['Name', 'MeritScore', 'Rank']].head(5)) pandas as pd
import numpy as np

np.random.seed(42)
n_students = 100
students = {
    'Name': [f'Student_{i+1}' for i in range(n_students)],
    'GPA': np.round(np.random.uniform(6.0, 10.0, n_students), 2),
    'EconomicStatus': np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.3, 0.4, 0.3]),
    'SchoolBoard': np.random.choice(['CBSE', 'ICSE', 'StateBoard'], n_students, p=[0.4, 0.3, 0.3]),
    'SetbackFlag': [True if i < 30 else False for i in range(n_students)],
    'InternshipExp': np.random.choice([True, False], n_students, p=[0.6, 0.4]),
    'ProjectsScore': np.round(np.random.uniform(0, 1, n_students), 2),
    'LeadershipScore': np.round(np.random.uniform(0, 1, n_students), 2),
    'MisdemeanorSeverity': np.random.choice([0.0, 0.5, 1.0], n_students, p=[0.96, 0.02, 0.02])
}

df = pd.DataFrame(students)
df.to_csv("student_data.csv", index=False)
import streamlit as st
import pandas as pd

st.title("FairRank: Merit-Based Hiring with Context")

import pandas as pd

# Use your actual raw GitHub CSV URL here
CSV_URL = "https://raw.githubusercontent.com/NG1897/Merit-Algorithm/main/student_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    GPA_WEIGHT = 0.50
    PROJECT_WEIGHT = 0.25
    LEADERSHIP_WEIGHT = 0.15
    INTERNSHIP_BONUS = 0.05

    def compute_merit(row):
        base_merit = (
            row['GPA'] * GPA_WEIGHT +
            row['ProjectsScore'] * PROJECT_WEIGHT +
            row['LeadershipScore'] * LEADERSHIP_WEIGHT +
            (INTERNSHIP_BONUS if row['InternshipExp'] else 0)
        )
        econ_boost = 0.10 if row['EconomicStatus'] == 'Low' else 0.05 if row['EconomicStatus'] == 'Medium' else 0.0
        board_boost = 0.05 if row['SchoolBoard'] == 'StateBoard' else 0.0
        setback_boost = 0.15 if row['SetbackFlag'] else 0.0
        total_boost = econ_boost + board_boost + setback_boost
        penalty = 0.05 * row['MisdemeanorSeverity']
        return base_merit * (1 + total_boost) * (1 - penalty)

    df['MeritScore'] = df.apply(compute_merit, axis=1)
    df['Rank'] = df['MeritScore'].rank(ascending=False, method='first').astype(int)
    df = df.sort_values(by='MeritScore', ascending=False)

    st.dataframe(df[['Name', 'GPA', 'EconomicStatus', 'SchoolBoard', 'SetbackFlag',
                     'InternshipExp', 'ProjectsScore', 'LeadershipScore', 'MisdemeanorSeverity',
                     'MeritScore', 'Rank']])
