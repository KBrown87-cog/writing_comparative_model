import streamlit as st
# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Rankings", layout="wide")

import pandas as pd
import numpy as np
from google.cloud.firestore_v1 import FieldFilter
from firebase_admin import firestore
from helpers import calculate_rankings, fetch_all_comparisons, save_rankings_to_firestore


st.title("📊 Ranked Writing Samples")

if "firestore_client" not in st.session_state:
    st.error("❌ Firestore is not initialized. Please go back to the main page and log in.")
    st.stop()

db = st.session_state["firestore_client"]
bucket = st.session_state["storage_bucket"]

# === BACK TO MAIN COMPARISON PAGE === #
if st.sidebar.button("🔙 Back to Judging"):
    st.switch_page("writing_analysis.py")  # Adjust if your main file has a different name

# === FETCH RANKINGS === #
@st.cache_data(ttl=60)
def fetch_ranked_images(school_name, year_group):
    """Fetch ranked writing samples with scores and teacher judgement."""
    try:
        docs = (
            st.session_state["firestore_client"].collection("rankings")
                .where(filter=FieldFilter("school", "==", school_name))
                .where(filter=FieldFilter("year_group", "==", year_group))
                .where(filter=FieldFilter("score", ">=", 0))
                .order_by("score", direction=firestore.Query.DESCENDING)
                .limit(50)
                .stream()
        )

        # Ensure we return all needed fields
        ranked = []
        for doc in docs:
            data = doc.to_dict()
            if "image_url" in data and "score" in data:
                # Include teacher judgement if available
                data["grade_label"] = data.get("grade_label", "Not Provided")
                ranked.append(data)

        return ranked

    except Exception as e:
        st.error(f"❌ Failed to fetch rankings: {str(e)}")
        return []


# === VALIDATE CONTEXT === #
if not st.session_state.get("logged_in") or not st.session_state.get("school_name"):
    st.warning("⚠️ Please log in from the main page first.")
    st.stop()

school_name = st.session_state["school_name"]
year_group = st.session_state.get("year_group", "Year 1")

ranked_images = fetch_ranked_images(school_name, year_group)

# === If rankings don't exist, calculate and save them ===
if not ranked_images:
    try:
        comparisons = st.session_state.get("comparisons")
        if not comparisons:
            comparisons = fetch_all_comparisons(school_name, year_group)

        rankings = calculate_rankings(comparisons)

        if rankings:
            save_rankings_to_firestore(rankings, school_name, year_group)
            ranked_images = fetch_ranked_images(school_name, year_group)

    except Exception as e:
        st.error(f"❌ Failed to calculate or fetch rankings: {str(e)}")

if ranked_images:
    # Convert to DataFrame
    df = pd.DataFrame(ranked_images)
    df = df.dropna(subset=["score"])

    # If scores are missing, skip categorization
    if df["score"].empty:
        st.warning("⚠️ Not enough rankings to categorize writing samples.")
        df["Standard"] = "Unranked"
    else:
        # Set cutoffs
        if len(df) < 10:
            min_score, max_score = df["score"].min(), df["score"].max()
            wts_cutoff = min_score - 1 if min_score == max_score else min_score + (max_score - min_score) * 0.3
            gds_cutoff = max_score + 1 if min_score == max_score else max_score - (max_score - min_score) * 0.3
        else:
            wts_cutoff = np.percentile(df["score"], 25)
            gds_cutoff = np.percentile(df["score"], 75)

        # Assign standard levels
        df["Standard"] = df["score"].apply(
            lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS")
        )

    # 🖼️ Thumbnail Table
    st.subheader("🖼️ Visual Ranking Table")

    # Convert ranked_images to DataFrame
    df = pd.DataFrame(ranked_images).dropna(subset=["score"])

    # Compute band if needed
    if df["score"].empty:
        st.warning("⚠️ Not enough rankings to categorize writing samples.")
        df["Standard"] = "Unranked"
    else:
        if len(df) < 10:
            min_score, max_score = df["score"].min(), df["score"].max()
            wts_cutoff = min_score + (max_score - min_score) * 0.3
            gds_cutoff = max_score - (max_score - min_score) * 0.3
        else:
            wts_cutoff = np.percentile(df["score"], 25)
            gds_cutoff = np.percentile(df["score"], 75)

        df["Standard"] = df["score"].apply(
            lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS")
        )

    # ✅ Safely create and rename columns
    df["Writing Sample"] = df["image_url"].apply(lambda url: f'<img src="{url}" width="480">')
    df["Teacher Judgement"] = df["grade_label"].fillna("Not Provided")
    df["Score"] = df["score"]
    df["Comparison Count"] = df["comparison_count"].fillna(0)
    df["Standard"] = df["Standard"]

    # ✅ CSV Download Button in Sidebar
    st.sidebar.download_button(
        label="📥 Download Rankings as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="writing_rankings.csv",
        mime="text/csv"
    )

    # ✅ Render the table
    st.markdown(
        df[["Writing Sample", "Score", "Comparison Count", "Teacher Judgement", "Standard"]]
        .to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    
