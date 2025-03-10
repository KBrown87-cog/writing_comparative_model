import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import random
import itertools
from scipy.optimize import minimize
import firebase_admin
from firebase_admin import credentials, firestore, storage

# === FIREBASE SETUP === #
import json
import io

# Load Firebase credentials from Streamlit Secrets
firebase_config_json = st.secrets["FIREBASE_KEY"]  # 🔹 Get the JSON as a string
firebase_config_dict = json.loads(firebase_config_json)  # 🔹 Convert string to dictionary

# Convert dictionary back to a file-like object (Firebase requires this format)
cred = credentials.Certificate(io.StringIO(json.dumps(firebase_config_dict)))

firebase_admin.initialize_app(cred, {
    'storageBucket': 'writing-comparison.firebasestorage.app'  # ✅ Ensure this matches your actual Firebase bucket
})

db = firestore.client()
bucket = storage.bucket()


# === STREAMLIT PAGE SETUP === #
st.set_page_config(layout="wide")
st.title("Comparative Judgement Writing Assessment")

# === SCHOOL LOGINS === #
SCHOOL_CREDENTIALS = {
    "School_A": hashlib.sha256("passwordA".encode()).hexdigest(),
    "School_B": hashlib.sha256("passwordB".encode()).hexdigest(),
    "adminkbrown": hashlib.sha256("115413Gtcs@".encode()).hexdigest()
}

# === SESSION STATE INITIALIZATION === #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.school_name = ""
    st.session_state.year_group = ""
    st.session_state.comparisons = []
    st.session_state.image_counts = {}
    st.session_state.scores = {}
    st.session_state.pairings = []

# === SIDEBAR LOGIN === #
school_name = st.sidebar.text_input("Enter School Name")
password = st.sidebar.text_input("Enter Password", type="password")
year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])
login_button = st.sidebar.button("Login")

if login_button:
    if school_name in SCHOOL_CREDENTIALS and hashlib.sha256(password.encode()).hexdigest() == SCHOOL_CREDENTIALS[school_name]:
        st.session_state.logged_in = True
        st.session_state.school_name = school_name
        st.session_state.year_group = year_group
        st.sidebar.success(f"Logged in as {school_name}, {year_group}")
    else:
        st.sidebar.error("Invalid credentials")

# === AFTER LOGIN === #
if st.session_state.logged_in:
    school_name = st.session_state.school_name
    year_group = st.session_state.year_group
    st.sidebar.header("Upload Writing Samples")

    uploaded_files = st.sidebar.file_uploader("Upload Writing Samples", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            blob = bucket.blob(f"{school_name}/{year_group}/{uploaded_file.name}")
            blob.upload_from_file(uploaded_file, content_type="image/jpeg")
            image_url = blob.public_url
            db.collection("writing_samples").add({
                "school": school_name,
                "year_group": year_group,
                "image_url": image_url,
                "filename": uploaded_file.name  # store filename for later deletion
            })
        st.success(f"{len(uploaded_files)} files uploaded!")

    # === DISPLAY + DELETE FILES === #
    st.subheader("Uploaded Samples")
    docs = db.collection("writing_samples").where("school", "==", school_name).where("year_group", "==", year_group).stream()
    image_docs = [doc for doc in docs]

    for doc in image_docs:
        data = doc.to_dict()
        st.image(data["image_url"], width=200, caption=data["filename"])
        if school_name == "adminkbrown":
            if st.button(f"🗑 Delete {data['filename']}", key=f"delete_{data['filename']}"):
                # Delete from Firebase Storage
                blob = bucket.blob(f"{school_name}/{year_group}/{data['filename']}")
                blob.delete()
                # Delete from Firestore
                db.collection("writing_samples").document(doc.id).delete()
                st.success(f"Deleted {data['filename']}")
                st.rerun()

    # === FETCH REMAINING IMAGES FOR COMPARISON === #
    docs = db.collection("writing_samples").where("school", "==", school_name).where("year_group", "==", year_group).stream()
    image_urls = [doc.to_dict()["image_url"] for doc in docs]

    if len(image_urls) >= 2:
        st.subheader("Vote for Your Favorite Image")
        st.write(f"Comparative Judgements: {len(st.session_state.comparisons)}")

        if not st.session_state.pairings:
            st.session_state.pairings = list(itertools.combinations(image_urls, 2))
            random.shuffle(st.session_state.pairings)

        def get_next_pair():
            return sorted(
                st.session_state.pairings,
                key=lambda p: st.session_state.image_counts.get(p[0], 0) + st.session_state.image_counts.get(p[1], 0)
            )[0]

        img1, img2 = get_next_pair()
        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, use_container_width=True)
            if st.button("Select this Image", key=f"vote_{img1}_{img2}"):
                st.session_state.comparisons.append((img1, img2, img1))
                st.session_state.image_counts[img1] = st.session_state.image_counts.get(img1, 0) + 1
                st.session_state.image_counts[img2] = st.session_state.image_counts.get(img2, 0) + 1
                st.session_state.scores[img1] = st.session_state.scores.get(img1, 0) + 1
                st.session_state.pairings.remove((img1, img2))
                st.rerun()

        with col2:
            st.image(img2, use_container_width=True)
            if st.button("Select this Image", key=f"vote_{img2}_{img1}"):
                st.session_state.comparisons.append((img1, img2, img2))
                st.session_state.image_counts[img1] = st.session_state.image_counts.get(img1, 0) + 1
                st.session_state.image_counts[img2] = st.session_state.image_counts.get(img2, 0) + 1
                st.session_state.scores[img2] = st.session_state.scores.get(img2, 0) + 1
                st.session_state.pairings.remove((img1, img2))
                st.rerun()


    # === RANKING SECTION === #
    def bradley_terry_log_likelihood(scores, comparisons):
        likelihood = 0
        for item1, item2, winner in comparisons:
            s1, s2 = scores[item1], scores[item2]
            p1 = np.exp(s1) / (np.exp(s1) + np.exp(s2))
            p2 = np.exp(s2) / (np.exp(s1) + np.exp(s2))
            likelihood += np.log(p1 if winner == item1 else p2)
        return -likelihood

    if st.session_state.comparisons:
        sample_names = list(set([item for sublist in st.session_state.comparisons for item in sublist[:2]]))
        initial_scores = {name: st.session_state.scores.get(name, 0) for name in sample_names}

        result = minimize(lambda s: bradley_terry_log_likelihood(dict(zip(sample_names, s)), st.session_state.comparisons),
                          list(initial_scores.values()), method='BFGS')

        final_scores = dict(zip(sample_names, result.x))
        ranked_samples = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(ranked_samples, columns=["Writing Sample", "Score"])
        wts_cutoff = np.percentile(df["Score"], 25)
        gds_cutoff = np.percentile(df["Score"], 75)
        df["Standard"] = df["Score"].apply(lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS"))

        st.subheader("Ranked Writing Samples")
        st.dataframe(df)
        st.sidebar.download_button("Download Results as CSV", df.to_csv(index=False).encode("utf-8"), "writing_rankings.csv", "text/csv")
