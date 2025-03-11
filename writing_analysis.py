import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import random
import itertools
from scipy.optimize import minimize
import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
import os

# ✅ Prevent duplicate Firebase initialization
if not firebase_admin._apps:
    firebase_config = {
        "type": st.secrets["FIREBASE"]["TYPE"],
        "project_id": st.secrets["FIREBASE"]["PROJECT_ID"],
        "private_key_id": st.secrets["FIREBASE"]["PRIVATE_KEY_ID"],
        "private_key": st.secrets["FIREBASE"]["PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": st.secrets["FIREBASE"]["CLIENT_EMAIL"],
        "client_id": st.secrets["FIREBASE"]["CLIENT_ID"],
        "auth_uri": st.secrets["FIREBASE"]["AUTH_URI"],
        "token_uri": st.secrets["FIREBASE"]["TOKEN_URI"],
        "auth_provider_x509_cert_url": st.secrets["FIREBASE"]["AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": st.secrets["FIREBASE"]["CLIENT_X509_CERT_URL"]
    }

    firebase_credentials_path = "/tmp/firebase_credentials.json"
    with open(firebase_credentials_path, "w") as json_file:
        json.dump(firebase_config, json_file)

    cred = credentials.Certificate(firebase_credentials_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'writing-comparison.firebasestorage.app'
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
password = st.sidebar.text_input("Enter Password", type="password", help="Case-sensitive")
login_button = st.sidebar.button("Login")

if login_button:
    if school_name in SCHOOL_CREDENTIALS and hashlib.sha256(password.encode()).hexdigest() == SCHOOL_CREDENTIALS[school_name]:
        st.session_state.logged_in = True
        st.session_state.school_name = school_name
        st.sidebar.success(f"Logged in as {school_name}")
    else:
        st.sidebar.error("Invalid credentials")

# === AFTER LOGIN === #
if st.session_state.logged_in:
    school_name = st.session_state.school_name
    st.sidebar.header("Select Year Group")
    year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])

    if year_group:
        st.session_state.year_group = year_group
        st.sidebar.header("Upload Writing Samples")

        uploaded_files = st.sidebar.file_uploader("Upload Writing Samples", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    blob = bucket.blob(f"{school_name}/{year_group}/{uploaded_file.name}")
                    blob.upload_from_file(uploaded_file, content_type="image/jpeg")
                    image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media"

                    db.collection("writing_samples").add({
                        "school": school_name,
                        "year_group": year_group,
                        "image_url": image_url,
                        "filename": uploaded_file.name
                    })

                    st.sidebar.success(f"{len(uploaded_files)} files uploaded successfully.")

                except Exception as e:
                    st.sidebar.error(f"❌ Upload Failed: {str(e)}")

        # === DISPLAY + DELETE FILES (Now in Sidebar) === #
        st.sidebar.header("Manage Uploaded Images")
        try:
            docs = db.collection("writing_samples")\
                     .where("school", "==", school_name)\
                     .where("year_group", "==", year_group)\
                     .stream()

            image_docs = [doc for doc in docs]

            if not image_docs:
                st.sidebar.warning("⚠️ No images found in Firestore! Check if Firestore is enabled and has data.")
            else:
                st.sidebar.success(f"✅ Found {len(image_docs)} images in Firestore.")

                for doc in image_docs:
                    data = doc.to_dict()
                    st.sidebar.image(data["image_url"], width=100, caption=data["filename"])

                    if school_name == "adminkbrown":
                        if st.sidebar.button(f"🗑 Delete {data['filename']}", key=f"delete_{doc.id}_{data['filename']}"):
                            try:
                                blob = bucket.blob(f"{school_name}/{year_group}/{data['filename']}")
                                blob.delete()
                                db.collection("writing_samples").document(doc.id).delete()

                                st.sidebar.success(f"Deleted {data['filename']}")
                                st.rerun()

                            except Exception as e:
                                st.sidebar.error(f"❌ Deletion Failed: {str(e)}")

        except Exception as e:
            st.sidebar.error(f"❌ Firestore Query Failed: {str(e)}")

# === PAGE: HOME (ONLY SHOW COMPARISON AND RANKINGS) === #
elif selected_option == "Home":
    st.title("Comparative Judgement Writing Assessment")
    st.write("Use the sidebar to navigate.")

    # ✅ Fetch images for comparison (DO NOT SHOW ALL IMAGES)
    image_urls = []
    for doc in image_docs:
        data = doc.to_dict()
        if "image_url" in data:
            image_urls.append(data["image_url"])

    # ✅ Ensure comparison logic only runs if we have at least 2 images
    if len(image_urls) >= 2:
        st.subheader("Vote for Your Favorite Image")
        st.write(f"Comparative Judgements: {len(st.session_state.comparisons)}")

        # ✅ Load existing rankings from Firestore for multi-user access
        def load_existing_comparisons(school_name, year_group):
            """Retrieve past rankings from Firestore."""
            try:
                docs = db.collection("rankings").where("school", "==", school_name)\
                                               .where("year_group", "==", year_group)\
                                               .stream()
                return [(doc.to_dict()["winning_image"], doc.to_dict()["losing_image"]) for doc in docs]
            except Exception as e:
                st.error(f"❌ Failed to fetch ranking data: {str(e)}")
                return []

        # ✅ Load previous rankings from Firestore when logging in
        if "pairings" not in st.session_state or not st.session_state.pairings:
            existing_comparisons = load_existing_comparisons(school_name, year_group)

            # ✅ Generate all possible pairs
            st.session_state.pairings = list(itertools.combinations(image_urls, 2))
            random.shuffle(st.session_state.pairings)

            # ✅ Remove already ranked pairs
            for comp in existing_comparisons:
                if comp in st.session_state.pairings:
                    st.session_state.pairings.remove(comp)

        def store_vote(winning_image, losing_image, school_name, year_group):
            """Stores the ranking vote in Firestore so it persists between logins."""
            try:
                db.collection("rankings").add({
                    "school": school_name,
                    "year_group": year_group,
                    "winning_image": winning_image,
                    "losing_image": losing_image,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                st.error(f"❌ Failed to record vote: {str(e)}")

        # ✅ Updated voting system
        next_pair = None
        if st.session_state.pairings:
            next_pair = sorted(
                st.session_state.pairings,
                key=lambda p: st.session_state.image_counts.get(p[0], 0) + st.session_state.image_counts.get(p[1], 0)
            )[0]

        if next_pair:
            img1, img2 = next_pair
            col1, col2 = st.columns(2)

            with col1:
                st.image(img1, use_container_width=True)
                if st.button("Select this Image", key=f"vote_{img1}_{img2}"):
                    st.session_state.pairings.remove((img1, img2))
                    store_vote(img1, img2, school_name, year_group)  # ✅ Store vote in Firestore
                    st.rerun()

            with col2:
                st.image(img2, use_container_width=True)
                if st.button("Select this Image", key=f"vote_{img2}_{img1}"):
                    st.session_state.pairings.remove((img1, img2))
                    store_vote(img2, img1, school_name, year_group)  # ✅ Store vote in Firestore
                    st.rerun()

# === RANKING SECTION === #
def bradley_terry_log_likelihood(scores, comparisons):
    """Calculates likelihood for Bradley-Terry ranking."""
    likelihood = 0
    for item1, item2, winner in comparisons:
        s1, s2 = scores[item1], scores[item2]
        p1 = np.exp(s1) / (np.exp(s1) + np.exp(s2))
        p2 = np.exp(s2) / (np.exp(s1) + np.exp(s2))
        likelihood += np.log(p1 if winner == item1 else p2)
    return -likelihood

# ✅ Fetch rankings from Firestore for report generation
def get_image_scores():
    """Fetches stored rankings and counts wins/losses for each image."""
    try:
        docs = db.collection("rankings").where("school", "==", school_name)\
                                       .where("year_group", "==", year_group)\
                                       .stream()
        scores = {}
        for doc in docs:
            vote = doc.to_dict()
            winner = vote.get("winning_image")  # ✅ Use `.get()` to avoid KeyError
            loser = vote.get("losing_image")

            if not winner or not loser:
                continue  # ✅ Skip invalid Firestore entries

            # Count wins
            scores.setdefault(winner, {"wins": 0, "losses": 0})["wins"] += 1
            # Count losses
            scores.setdefault(loser, {"wins": 0, "losses": 0})["losses"] += 1

        return scores
    except Exception as e:
        st.error(f"❌ Failed to fetch ranking data: {str(e)}")
        return {}

# ✅ Fetch all stored comparisons from Firestore for rankings
def fetch_all_comparisons(school_name, year_group):
    """Retrieves all stored rankings from Firestore and ensures correct format."""
    try:
        docs = db.collection("rankings").where("school", "==", school_name)\
                                       .where("year_group", "==", year_group)\
                                       .stream()
        comparisons = []
        for doc in docs:
            data = doc.to_dict()
            winner = data.get("winning_image")
            loser = data.get("losing_image")
            if winner and loser:
                comparisons.append((winner, loser, winner))  # ✅ Ensure tuple has 3 elements


        return comparisons
    except Exception as e:
        st.error(f"❌ Failed to fetch comparison data: {str(e)}")
        return []

# ✅ Load rankings from Firestore
stored_comparisons = fetch_all_comparisons(school_name, year_group)

if "comparisons" not in st.session_state:
    st.session_state.comparisons = []

# ✅ Prevent duplicate votes from being re-added
for comp in stored_comparisons:
    if comp not in st.session_state.comparisons:
        st.session_state.comparisons.append(comp)


# ✅ Prevent running minimize() if no comparisons exist
if not st.session_state.comparisons or any(len(comp) != 3 for comp in st.session_state.comparisons):
    st.warning("⚠️ No valid comparisons available. Ranking cannot be calculated yet.")
else:
    sample_names = list(set([item for sublist in st.session_state.comparisons for item in sublist[:2]]))
    
    # ✅ Initialize scores for all sample names
    initial_scores = {name: st.session_state.scores.get(name, 0) for name in sample_names}
    for name in sample_names:
        if name not in initial_scores:
            initial_scores[name] = 0  # ✅ Ensure all samples have a score


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
