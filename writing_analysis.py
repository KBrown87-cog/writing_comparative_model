import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import os
import random
import itertools
from PIL import Image
from scipy.optimize import minimize

# 📌 Ensure persistent storage
BASE_DIR = "writing_comparative_model"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# 🔹 Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("Comparative Judgement Writing Assessment")

# 🔹 Authentication system
SCHOOL_CREDENTIALS = {
    "School_A": hashlib.sha256("passwordA".encode()).hexdigest(),
    "School_B": hashlib.sha256("passwordB".encode()).hexdigest(),
    "adminkbrown": hashlib.sha256("115413Gtcs@".encode()).hexdigest()
}

# 🔹 Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.school_name = ""
    st.session_state.year_group = ""
    st.session_state.comparisons = []
    st.session_state.image_counts = {}
    st.session_state.scores = {}
    st.session_state.writing_samples = {}
    st.session_state.pairings = []
    st.session_state.rankings_df = pd.DataFrame()

# 🔹 Login System
school_name = st.sidebar.text_input("Enter School Name")
password = st.sidebar.text_input("Enter Password", type="password")
login_button = st.sidebar.button("Login")
logout_button = st.sidebar.button("Logout")

if login_button:
    if school_name in SCHOOL_CREDENTIALS and hashlib.sha256(password.encode()).hexdigest() == SCHOOL_CREDENTIALS[school_name]:
        st.sidebar.success(f"Logged in as {school_name}")
        st.session_state.logged_in = True
        st.session_state.school_name = school_name
    else:
        st.sidebar.error("Invalid credentials. Please try again.")

if logout_button:
    st.session_state.logged_in = False
    st.session_state.school_name = ""
    st.session_state.year_group = ""
    st.rerun()  # ✅ FIXED! Replaces `st.experimental_rerun()`

# 🔹 Year Group Selection
if st.session_state.logged_in:
    year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])
    st.session_state.year_group = year_group

    # Set up year group folder
    YEAR_GROUP_FOLDER = os.path.join(UPLOAD_FOLDER, school_name, year_group)
    os.makedirs(YEAR_GROUP_FOLDER, exist_ok=True)

    # File paths for data storage
    comparisons_file = os.path.join(DATA_FOLDER, f"{school_name}_{year_group}_comparisons.csv")
    rankings_file = os.path.join(DATA_FOLDER, f"{school_name}_{year_group}_rankings.csv")

    # 🔹 Load previous data
    if os.path.exists(comparisons_file):
        st.session_state.comparisons = pd.read_csv(comparisons_file).values.tolist()
    if os.path.exists(rankings_file):
        st.session_state.rankings_df = pd.read_csv(rankings_file)

    # 🔹 Upload Writing Samples
    uploaded_files = st.sidebar.file_uploader("Upload Writing Samples", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(YEAR_GROUP_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_path not in st.session_state.writing_samples.get(year_group, []):
                st.session_state.writing_samples.setdefault(year_group, []).append(file_path)
                st.session_state.image_counts[file_path] = 0
                st.session_state.scores[file_path] = 0

    # 🔹 Comparison System
    if len(st.session_state.writing_samples.get(year_group, [])) >= 2:
        st.subheader("Vote for Your Favorite Image")
        st.write(f"Comparative Judgements: {len(st.session_state.comparisons)}")

        # **Balanced Comparisons**
        sample_images = st.session_state.writing_samples[year_group]
        if not st.session_state.pairings:
            st.session_state.pairings = list(itertools.combinations(sample_images, 2))
            random.shuffle(st.session_state.pairings)

        def get_next_pair():
            sorted_pairs = sorted(st.session_state.pairings, key=lambda p: st.session_state.image_counts[p[0]] + st.session_state.image_counts[p[1]])
            return sorted_pairs[0]

        img1, img2 = get_next_pair()

        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, use_column_width=True)
            if st.button("Select", key=f"vote_{img1}_{img2}"):
                st.session_state.comparisons.append((img1, img2, img1))
                st.session_state.image_counts[img1] += 1
                st.session_state.image_counts[img2] += 1
                st.session_state.pairings.remove((img1, img2))
                st.rerun()

        with col2:
            st.image(img2, use_column_width=True)
            if st.button("Select", key=f"vote_{img2}_{img1}"):
                st.session_state.comparisons.append((img1, img2, img2))
                st.session_state.image_counts[img1] += 1
                st.session_state.image_counts[img2] += 1
                st.session_state.pairings.remove((img1, img2))
                st.rerun()

    # 🔹 Ranking System (Bradley-Terry Model)
    if st.session_state.comparisons:
        sample_names = list(set([item for sublist in st.session_state.comparisons for item in sublist[:2]]))
        initial_scores = {name: st.session_state.scores.get(name, 0) for name in sample_names}

        def bradley_terry_log_likelihood(scores, comparisons):
            likelihood = 0
            for item1, item2, winner in comparisons:
                s1, s2 = scores[item1], scores[item2]
                p1 = np.exp(s1) / (np.exp(s1) + np.exp(s2))
                p2 = np.exp(s2) / (np.exp(s1) + np.exp(s2))
                likelihood += np.log(p1 if winner == item1 else p2)
            return -likelihood

        result = minimize(lambda s: bradley_terry_log_likelihood(dict(zip(sample_names, s)), st.session_state.comparisons),
                          list(initial_scores.values()), method='BFGS')
        final_scores = dict(zip(sample_names, result.x))

        df = pd.DataFrame(final_scores.items(), columns=["Writing Sample", "Score"])
        
        # **Apply Standardized Ranking**
        wts_cutoff = np.percentile(list(final_scores.values()), 25)
        gds_cutoff = np.percentile(list(final_scores.values()), 75)
        df["Standard"] = df["Score"].apply(lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS"))

        df.to_csv(rankings_file, index=False)

        st.sidebar.download_button("Download Year Group Data", df.to_csv(index=False).encode("utf-8"), "year_group_rankings.csv", "text/csv")

        st.subheader("Ranked Writing Samples")
        st.dataframe(df)
