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

        # === DISPLAY + DELETE FILES (In Sidebar) === #
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
# ✅ Fetch images from Firestore to compare
image_urls = []
try:
    docs = db.collection("writing_samples")\
             .where("school", "==", school_name)\
             .where("year_group", "==", year_group)\
             .stream()

    for doc in docs:
        data = doc.to_dict()
        if "image_url" in data:
            image_urls.append(data["image_url"])  # ✅ Collect valid image URLs

except Exception as e:
    st.error(f"❌ Firestore Query Failed: {str(e)}")

# ✅ Prevent error if no images exist
if not image_urls:
    st.warning("⚠️ No images found in Firestore. Upload images to start comparisons.")
    st.stop()  # ✅ Stops execution to prevent errors

# ✅ Display images for comparison
if len(image_urls) >= 2:
    st.subheader("Vote for Your Favorite Image")
    st.write(f"Comparative Judgements: {len(st.session_state.comparisons)}")

    if "pairings" not in st.session_state or not st.session_state.pairings:
        st.session_state.pairings = list(itertools.combinations(image_urls, 2))
        random.shuffle(st.session_state.pairings)

    if st.session_state.pairings:
        img1, img2 = st.session_state.pairings.pop(0)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, use_container_width=True)
            if st.button("Select this Image", key=f"vote_{img1}_{img2}"):
                store_vote(img1, img2, school_name, year_group)  # ✅ Store vote in Firestore
                st.rerun()

        with col2:
            st.image(img2, use_container_width=True)
            if st.button("Select this Image", key=f"vote_{img2}_{img1}"):
                store_vote(img2, img1, school_name, year_group)  # ✅ Store vote in Firestore
                st.rerun()


    else:
        st.warning("⚠️ No more image pairs available for comparison. Upload more images to continue voting.")

# === RANKING SYSTEM: USING BRADLEY-TERRY MODEL === #
def bradley_terry_log_likelihood(scores, comparisons):
    """Calculates likelihood for Bradley-Terry ranking."""
    likelihood = 0
    for item1, item2, winner in comparisons:
        s1, s2 = scores.get(item1, 0), scores.get(item2, 0)  # Default scores to 0
        p1 = np.exp(s1) / (np.exp(s1) + np.exp(s2))
        p2 = np.exp(s2) / (np.exp(s1) + np.exp(s2))
        likelihood += np.log(p1 if winner == item1 else p2)
    return -likelihood

# ✅ Store votes in Firestore
def store_vote(winning_image, losing_image, school_name, year_group):
    """Stores the vote result in Firestore and updates ranking scores."""
    try:
        # References for the images
        winning_ref = db.collection("rankings").document(winning_image)
        losing_ref = db.collection("rankings").document(losing_image)

        # Get existing data (if any)
        winning_doc = winning_ref.get()
        losing_doc = losing_ref.get()

        # Fetch existing scores or initialize them
        winning_score = winning_doc.to_dict().get("score", 0) if winning_doc.exists else 0
        losing_score = losing_doc.to_dict().get("score", 0) if losing_doc.exists else 0

        # Update the scores
        winning_score += 1.2 / (1 + winning_score)  # Reward the selected image
        losing_score -= 0.8 / (1 + losing_score)    # Penalize the non-selected image

        # Update Firestore with the new scores
        winning_ref.set({
            "school": school_name,
            "year_group": year_group,
            "image_url": winning_image,
            "score": winning_score
        }, merge=True)

        losing_ref.set({
            "school": school_name,
            "year_group": year_group,
            "image_url": losing_image,
            "score": losing_score
        }, merge=True)

    except Exception as e:
        st.error(f"❌ Failed to update ranking: {str(e)}")



# ✅ Fetch image rankings from Firestore
def fetch_ranked_images(school_name, year_group):
    """Fetches all ranked images from Firestore and sorts them by score."""
    try:
        docs = db.collection("rankings").where("school", "==", school_name)\
                                       .where("year_group", "==", year_group)\
                                       .stream()
        scores = []
        for doc in docs:
            data = doc.to_dict()
            scores.append((data["image_url"], data.get("score", 0), data.get("votes", 0)))

        # ✅ Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)

    except Exception as e:
        st.error(f"❌ Failed to fetch ranked images: {str(e)}")
        return []

# === DISPLAY FINAL RANKINGS === #
st.subheader("Ranked Writing Samples")

# ✅ Fetch final rankings from Firestore
ranked_images = fetch_ranked_images(school_name, year_group)

if ranked_images:
    df = pd.DataFrame(ranked_images, columns=["Writing Sample", "Score"])

    # ✅ Apply GDS, EXS, WTS thresholds based on percentiles
    wts_cutoff = np.percentile(df["Score"], 25)
    gds_cutoff = np.percentile(df["Score"], 75)
    df["Standard"] = df["Score"].apply(lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS"))

    st.dataframe(df)
    st.sidebar.download_button("Download Results as CSV", df.to_csv(index=False).encode("utf-8"), "writing_rankings.csv", "text/csv")
else:
    st.warning("⚠️ No ranked images found. Begin voting to generate rankings.")
