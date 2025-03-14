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

    # ✅ Detect Year Group Change
    previous_year_group = st.session_state.get("year_group", None)
    year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])

    if year_group != previous_year_group:
        # ✅ Reset session state when switching year groups
        st.session_state.year_group = year_group
        st.session_state.image_urls = []
        st.session_state.pairings = []
        st.session_state.comparisons = []
        st.session_state.rankings = []
        st.session_state.uploaded_files = []  # ✅ Reset uploaded files list

        # ✅ Immediately fetch images for the new year group
        docs = db.collection("writing_samples")\
                 .where("school", "==", school_name)\
                 .where("year_group", "==", year_group)\
                 .stream()

        st.session_state.image_urls = [doc.to_dict()["image_url"] for doc in docs]

        st.rerun()  # ✅ Ensures full refresh

    # ✅ UPLOAD WRITING SAMPLES
    st.sidebar.header("Upload Writing Samples")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Writing Samples", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        # ✅ Clear previous uploaded files when switching year group
        if "uploaded_files" in st.session_state and year_group != previous_year_group:
            st.session_state.uploaded_files = []

        for uploaded_file in uploaded_files:
            try:
                # ✅ Ensure correct year group is selected
                year_group = st.session_state.get("year_group", None)
                if not year_group:
                    st.error("⚠️ Please select a year group before uploading images.")
                    st.stop()

                # ✅ Upload to Firebase Storage
                blob = bucket.blob(f"{school_name}/{year_group}/{uploaded_file.name}")
                blob.upload_from_file(uploaded_file, content_type="image/jpeg")
                image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media"

                # ✅ Store in Firestore
                db.collection("writing_samples").add({
                    "school": school_name,
                    "year_group": year_group,
                    "image_url": image_url,
                    "filename": uploaded_file.name
                })

                st.session_state.uploaded_files.append(image_url)  # ✅ Immediately add new image to session
                st.sidebar.success(f"{len(uploaded_files)} files uploaded successfully.")

            except Exception as e:
                st.sidebar.error(f"❌ Upload Failed: {str(e)}")


    # ✅ DISPLAY + DELETE FILES (In Sidebar)
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


# === DISPLAY VOTING IMAGES ABOVE RANKINGS === #
# ✅ Fetch images from Firestore for the selected year group
image_urls = []
try:
    docs = db.collection("writing_samples") \
             .where("school", "==", school_name) \
             .where("year_group", "==", year_group) \
             .stream()  # ✅ Removed the incorrect trailing backslash

    for doc in docs:
        data = doc.to_dict()
        if "image_url" in data:
            image_urls.append(data["image_url"])  # ✅ Ensure only selected year group images are collected

except Exception as e:
    st.error(f"❌ Firestore Query Failed: {str(e)}")


# ✅ Prevent error if no images exist for the selected year group
if not image_urls:
    st.warning("⚠️ No images found for the selected year group. Upload images to start comparisons.")
    st.stop()  

# ✅ Ensure new images are presented for voting
if len(st.session_state.image_urls) >= 2:
    st.subheader(f"Comparing Writing Samples for {year_group}")

    if "pairings" not in st.session_state or not st.session_state.pairings:
        st.session_state.pairings = list(itertools.combinations(st.session_state.image_urls, 2))
        random.shuffle(st.session_state.pairings)

    # ✅ Process each pair one by one
    if st.session_state.pairings:
        img1, img2 = st.session_state.pairings.pop(0)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, use_container_width=True)
            if st.button(f"Select {img1}", key=f"vote_{hash(img1)}_{hash(img2)}"):
                store_vote(img1, img2, school_name, year_group)
                st.rerun()

        with col2:
            st.image(img2, use_container_width=True)
            if st.button(f"Select {img2}", key=f"vote_{hash(img2)}_{hash(img1)}"):
                store_vote(img2, img1, school_name, year_group)
                st.rerun()

        # ✅ Automatically store the comparison in Firestore
        try:
            db.collection("comparisons").add({
                "school": school_name,
                "year_group": year_group,
                "image_1": img1,
                "image_2": img2,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            st.success(f"Comparison stored for {year_group}")
        except Exception as e:
            st.error(f"❌ Failed to store comparison: {str(e)}")

    else:
        st.warning("⚠️ No more image pairs available for comparison. Upload more images to continue.")


# ✅ Store votes in Firestore correctly
def store_vote(selected_image, other_image, school_name, year_group):
    """Stores votes and updates ranking scores in Firestore."""
    try:
        # ✅ Generate valid Firestore document IDs
        selected_doc_id = hashlib.sha256(selected_image.encode()).hexdigest()[:20]
        other_doc_id = hashlib.sha256(other_image.encode()).hexdigest()[:20]

        selected_ref = db.collection("rankings").document(selected_doc_id)
        other_ref = db.collection("rankings").document(other_doc_id)

        selected_doc = selected_ref.get()
        other_doc = other_ref.get()

        # ✅ Get existing scores or initialize if not found
        selected_data = selected_doc.to_dict() if selected_doc.exists else {"score": 0, "votes": 0}
        other_data = other_doc.to_dict() if other_doc.exists else {"score": 0, "votes": 0}

        # ✅ Extract previous values
        selected_score = selected_data.get("score", 0)
        other_score = other_data.get("score", 0)
        selected_votes = selected_data.get("votes", 0)
        other_votes = other_data.get("votes", 0)

        # ✅ Update Scores & Votes
        selected_score += 1.2 / (1 + selected_score)  
        other_score -= 0.8 / (1 + other_score)  
        selected_votes += 1  
        other_votes += 1  

        # ✅ Update Firestore with new values
        selected_ref.set({
            "school": school_name,
            "year_group": year_group,
            "image_url": selected_image,
            "score": selected_score,
            "votes": selected_votes
        }, merge=True)

        other_ref.set({
            "school": school_name,
            "year_group": year_group,
            "image_url": other_image,
            "score": other_score,
            "votes": other_votes
        }, merge=True)

    except Exception as e:
        st.error(f"❌ Failed to update image scores: {str(e)}")


# ✅ Fetch all stored comparisons from Firestore
def fetch_all_comparisons(school_name, year_group):
    """Retrieves all stored comparisons from Firestore for the selected year group."""
    try:
        docs = (
            db.collection("comparisons")
            .where("school", "==", school_name)
            .where("year_group", "==", year_group)  # ✅ Ensure only selected year group
            .stream()  # ✅ Fix: Correct indentation and remove unnecessary backslash
        )

        comparisons = []
        for doc in docs:
            data = doc.to_dict()
            img1 = data.get("image_1")
            img2 = data.get("image_2")
            winner = data.get("winner", img1)  # ✅ Default to img1 if no explicit winner stored

            if img1 and img2:
                comparisons.append((img1, img2, winner))  # ✅ Store proper format

        if not comparisons:
            st.info("ℹ️ No comparisons found for the selected year group yet. Start making comparisons!")

        return comparisons
    except Exception as e:
        st.error(f"❌ Failed to fetch comparison data: {str(e)}")
        return []



# ✅ Calculate Rankings Using Bradley-Terry Model
def calculate_rankings(comparisons):
    """Applies Bradley-Terry Model to rank images."""
    if not comparisons:
        st.warning("⚠️ No valid comparisons available. Ranking cannot be calculated yet.")
        return {}

    # ✅ Extract unique image names
    sample_names = list(set([img for pair in comparisons for img in pair[:2]]))  # Ensure only image names
    initial_scores = {name: 0 for name in sample_names}

    try:
        result = minimize(lambda s: bradley_terry_log_likelihood(dict(zip(sample_names, s)), comparisons),
                          list(initial_scores.values()), method='BFGS')

        return dict(zip(sample_names, result.x))
    except Exception as e:
        st.error(f"❌ Ranking Calculation Failed: {str(e)}")
        return {}



# ✅ Bradley-Terry Model for Ranking
def bradley_terry_log_likelihood(scores, comparisons):
    """Calculates likelihood for Bradley-Terry ranking."""
    likelihood = 0

    if not comparisons:
        st.error("❌ No comparisons received in Bradley-Terry function.")
        return float('inf')

    for item1, item2, winner in comparisons:
        s1, s2 = scores.get(item1, 0), scores.get(item2, 0)  # Default to 0
        p1 = np.exp(s1) / (np.exp(s1) + np.exp(s2))
        p2 = np.exp(s2) / (np.exp(s1) + np.exp(s2))
        likelihood += np.log(p1 if winner == item1 else p2)

    return -likelihood

# ✅ Fetch Rankings from Firestore and Apply Bradley-Terry Model
stored_comparisons = fetch_all_comparisons(school_name, year_group)

if stored_comparisons:
    rankings = calculate_rankings(stored_comparisons)

    # ✅ Store Rankings in Firestore
    for image, score in rankings.items():
        # ✅ Ensure image URL is sanitized before using it as a Firestore document ID
        doc_id = hashlib.sha256(image.encode()).hexdigest()[:20]  # Create a short, unique ID

        db.collection("rankings").document(doc_id).set({
            "school": school_name,
            "year_group": year_group,
            "image_url": image,  # Store full image URL inside the document
            "score": float(score)  # ✅ Store as float for consistency
        }, merge=True)

# ✅ Define function to fetch ranked images before calling it
def fetch_ranked_images(school_name, year_group):
    """Fetches all ranked images from Firestore and sorts them by score."""
    try:
        docs = db.collection("rankings")\
                 .where("school", "==", school_name)\
                 .where("year_group", "==", year_group)\
                 .stream()

        scores = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("year_group") == year_group:  # ✅ Ensure only selected year group rankings appear
                scores.append((data["image_url"], data.get("score", 0), data.get("votes", 0)))

        # ✅ Sort images by score (higher score = better ranking)
        return sorted(scores, key=lambda x: x[1], reverse=True)

    except Exception as e:
        st.error(f"❌ Failed to fetch ranked images: {str(e)}")
        return []



# ✅ Now call `fetch_ranked_images` at the correct location
ranked_images = fetch_ranked_images(school_name, year_group)

# ✅ Display Rankings in a Table
if ranked_images:
    df = pd.DataFrame(ranked_images, columns=["Writing Sample", "Score", "Votes"])
    
    # ✅ Apply GDS, EXS, WTS thresholds based on percentiles
    wts_cutoff = np.percentile(df["Score"], 25)
    gds_cutoff = np.percentile(df["Score"], 75)
    df["Standard"] = df["Score"].apply(lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS"))

    st.subheader("Ranked Writing Samples")
    st.dataframe(df)
    st.sidebar.download_button("Download Results as CSV", df.to_csv(index=False).encode("utf-8"), "writing_rankings.csv", "text/csv")
else:
    st.warning("⚠️ No ranked images found for this year group. Begin voting to generate rankings.")
