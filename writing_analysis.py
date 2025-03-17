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

# ✅ Debug Mode Toggle (set to False for normal use, True for debugging)
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ✅ Prevent duplicate Firebase initialization
if "firebase_initialized" not in st.session_state:
    st.session_state.firebase_initialized = True
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
    st.session_state.image_urls = []
    st.session_state.pairings = []
    st.session_state.comparisons = []
    st.session_state.rankings = []
    st.session_state.image_comparison_counts = {}

if not st.session_state.logged_in:
    st.sidebar.header("Login")
    school_name = st.sidebar.text_input("Enter School Name")
    password = st.sidebar.text_input("Enter Password", type="password", help="Case-sensitive")

    if st.sidebar.button("Login"):
        school_name = school_name.strip()
        if school_name in SCHOOL_CREDENTIALS and hashlib.sha256(password.encode()).hexdigest() == SCHOOL_CREDENTIALS[school_name]:
            st.session_state.logged_in = True
            st.session_state.school_name = school_name
            st.sidebar.success(f"Logged in as {school_name}")
        else:
            st.sidebar.error("Invalid credentials. Please check your username and password.")

else:
    st.sidebar.header(f"Logged in as {st.session_state.school_name}")
    if st.sidebar.button("Logout"):
        keys_to_clear = ["logged_in", "school_name", "year_group"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

# === AFTER LOGIN === #
if st.session_state.logged_in:
    school_name = st.session_state.school_name
    st.sidebar.header("Select Year Group")
    year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])

    if year_group != st.session_state.year_group:
        st.session_state.year_group = year_group
        st.session_state.image_urls = []
        st.session_state.image_comparison_counts = {}
        docs = db.collection("writing_samples")\
                 .where("school", "==", school_name)\
                 .where("year_group", "==", year_group)\
                 .stream()

        image_pool = {"GDS": [], "EXS": [], "WTS": []}
        st.session_state.image_urls = []

        for doc in docs:
            data = doc.to_dict()
            if "image_url" in data and "grade_label" in data:
                image_pool[data["grade_label"]].append(data["image_url"])
                st.session_state.image_urls.append(data["image_url"])
                st.session_state.image_comparison_counts[data["image_url"]] = 0

    # ✅ Prevent errors if no images found
    if not any(image_pool.values()):
        st.warning("⚠️ No images available for comparison. Upload images first.")
        st.stop()

    # ✅ Ensure fair sample distribution across GDS, EXS, WTS
    sample_pool = {"GDS": [], "EXS": [], "WTS": []}

    for img_url in st.session_state.image_urls:
        for grade, images in image_pool.items():
            if img_url in images:
                sample_pool[grade].append(img_url)

    # ✅ Generate pairs from mixed ability levels
    all_pairs = []
    for grade, images in sample_pool.items():
        random.shuffle(images)
        for pair in itertools.combinations(images, 2):
            all_pairs.append((pair[0], pair[1], grade))

    random.shuffle(all_pairs)

    # ✅ Prioritize images with fewer comparisons
    all_pairs.sort(key=lambda pair: (
        st.session_state.image_comparison_counts.get(pair[0], 0) +
        st.session_state.image_comparison_counts.get(pair[1], 0)
    ))

    num_pairs = min(40, len(all_pairs))  # ✅ Increase fairness (adjust as needed)
    st.session_state.pairings = random.sample(all_pairs, num_pairs)

    # ✅ Process one pair at a time using click-to-select
    if st.session_state.pairings:
        img1, img2, _ = st.session_state.pairings.pop(0)

        col1, col2 = st.columns(2)

        if col1.image(img1, use_container_width=True) and col1.button("Select", key=f"vote_{img1}_{img2}"):
            store_vote(img1, img2, school_name, year_group)
            st.rerun()

        if col2.image(img2, use_container_width=True) and col2.button("Select", key=f"vote_{img2}_{img1}"):
            store_vote(img2, img1, school_name, year_group)
            st.rerun()

    try:
        # ✅ Ensure last selected image exists before assigning winner
        if "last_selected" not in st.session_state:
            st.error("❌ No selection recorded. Please select an image first.")
            return

        # ✅ Determine winner based on user selection
        winner = img1 if st.session_state.last_selected == img1 else img2

        # ✅ Store comparison in Firestore
        db.collection("comparisons").add({
            "school": school_name,
            "year_group": year_group,
            "image_1": img1,
            "image_2": img2,
            "winner": winner,  # ✅ Track which image was selected
            "timestamp": firestore.SERVER_TIMESTAMP,
            "comparison_count": firestore.Increment(1)  # ✅ Track number of comparisons
        }, merge=True)  # ✅ Prevents duplicate entries
    except Exception as e:
        st.error(f"❌ Failed to store comparison: {str(e)}")



      
def store_vote(selected_image, other_image, school_name, year_group):
    """Stores votes and updates ranking scores in Firestore using Firestore Transactions."""

    # ✅ Prevent self-comparison issues
    if selected_image == other_image:
        st.error("❌ Invalid comparison: Both images are the same.")
        return

    try:
        # ✅ Generate Firestore document IDs
        selected_doc_id = hashlib.sha256(selected_image.encode()).hexdigest()[:20]
        other_doc_id = hashlib.sha256(other_image.encode()).hexdigest()[:20]

        selected_ref = db.collection("rankings").document(selected_doc_id)
        other_ref = db.collection("rankings").document(other_doc_id)

        def transaction_update(transaction):
            # ✅ Get current data in one transaction (avoiding multiple `.get()` calls)
            selected_doc = selected_ref.get(transaction=transaction)
            other_doc = other_ref.get(transaction=transaction)

            # ✅ Get existing scores or initialize if not found
            selected_data = selected_doc.to_dict() if selected_doc.exists else {"score": 0, "votes": 0, "comparison_count": 0}
            other_data = other_doc.to_dict() if other_doc.exists else {"score": 0, "votes": 0, "comparison_count": 0}

            # ✅ Extract previous values
            selected_score = selected_data.get("score", 0)
            other_score = other_data.get("score", 0)
            selected_votes = selected_data.get("votes", 0)
            other_votes = other_data.get("votes", 0)
            selected_comparisons = selected_data.get("comparison_count", 0)
            other_comparisons = other_data.get("comparison_count", 0)

            # ✅ Apply Bradley-Terry Model Update
            K = 1.0  # ✅ Scaling Factor
            expected_score = 1 / (1 + np.exp(other_score - selected_score))  # Expected probability

            selected_score += K * (1 - expected_score)  # ✅ Adjust for winner
            other_score -= K * expected_score  # ✅ Adjust for loser

            # ✅ Track number of times each image has been compared
            selected_votes += 1
            other_votes += 1
            selected_comparisons += 1
            other_comparisons += 1

            # ✅ Update Firestore in one transaction
            transaction.set(selected_ref, {
                "school": school_name,
                "year_group": year_group,
                "image_url": selected_image,
                "score": selected_score,
                "votes": selected_votes,
                "comparison_count": selected_comparisons
            }, merge=True)

            transaction.set(other_ref, {
                "school": school_name,
                "year_group": year_group,
                "image_url": other_image,
                "score": other_score,
                "votes": other_votes,
                "comparison_count": other_comparisons
            }, merge=True)

        db.run_transaction(transaction_update)

    except Exception as e:
        st.error(f"❌ Failed to update image scores: {str(e)}")



# ✅ Fetch all stored comparisons from Firestore
def fetch_all_comparisons(school_name, year_group):
    """Retrieves all stored comparisons from Firestore for the selected year group, ordered by most recent."""
    try:
        docs = (
            db.collection("comparisons")
            .where("school", "==", school_name)
            .where("year_group", "==", year_group)  # ✅ Ensure only selected year group
            .order_by("timestamp", direction=firestore.Query.DESCENDING)  # ✅ Sort by most recent comparisons
            .limit(100)  # ✅ Prevent excessive reads (adjust as needed)
            .stream()
        )

        comparisons = []
        for doc in docs:
            data = doc.to_dict()

            img1 = data.get("image_1")
            img2 = data.get("image_2")
            winner = data.get("winner", img1)  # ✅ Default to img1 if winner is missing
            comparison_count = data.get("comparison_count", 0)  # ✅ Track how many times this pair was compared

            # ✅ Ensure data integrity before adding to the list
            if img1 and img2:
                comparisons.append((img1, img2, winner, comparison_count))

        if not comparisons:
            st.info("ℹ️ No comparisons found for the selected year group yet. Start making comparisons!")

        return comparisons
    except Exception as e:
        st.error(f"❌ Failed to fetch comparison data: {str(e)}")
        return []



# ✅ Calculate Rankings Using Bradley-Terry Model
def calculate_rankings(comparisons):
    """Applies Bradley-Terry Model to rank images, incorporating weighting and convergence checks."""
    if not comparisons:
        st.warning("⚠️ No valid comparisons available. Ranking cannot be calculated yet.")
        return {}

    # ✅ Extract unique image names and count how many times each image was compared
    comparison_counts = {}
    for img1, img2, winner, _ in comparisons:  # Extract comparison count
        comparison_counts[img1] = comparison_counts.get(img1, 0) + 1
        comparison_counts[img2] = comparison_counts.get(img2, 0) + 1

    # ✅ Remove images that were never compared
    sample_names = [name for name, count in comparison_counts.items() if count > 0]

    if not sample_names:
        st.warning("⚠️ No valid image comparisons available for ranking.")
        return {}

    # ✅ Initialize scores with small random values to improve convergence
    initial_scores = {name: np.random.uniform(-0.1, 0.1) for name in sample_names}

    try:
        # ✅ Perform ranking optimization with weightings based on comparison counts
        result = minimize(
            lambda s: bradley_terry_log_likelihood(dict(zip(sample_names, s)), comparisons, comparison_counts),
            list(initial_scores.values()), 
            method='BFGS'
        )

        # ✅ Check if optimization succeeded
        if not result.success:
            st.warning("⚠️ Optimization failed to converge. Using default scores.")
            return {name: 0 for name in sample_names}  # Return neutral scores if failure

        return dict(zip(sample_names, result.x))

    except Exception as e:
        st.error(f"❌ Ranking Calculation Failed: {str(e)}")
        return {}

# ✅ Bradley-Terry Model for Ranking with Weighting
def bradley_terry_log_likelihood(scores, comparisons):
    """Calculates likelihood for Bradley-Terry ranking with weighting."""
    likelihood = 0

    if not comparisons:
        st.error("❌ No comparisons received in Bradley-Terry function.")
        return float('inf')

    for img1, img2, winner, comparison_count in comparisons:
        s1, s2 = scores.get(img1, 0), scores.get(img2, 0)  # Default scores
        exp_s1, exp_s2 = np.exp(s1), np.exp(s2)

        # ✅ Compute win probabilities with log safety
        p1 = max(exp_s1 / (exp_s1 + exp_s2), 1e-10)
        p2 = max(exp_s2 / (exp_s1 + exp_s2), 1e-10)

        # ✅ Apply weighting to ensure fair impact from multiple comparisons
        weight = np.log(comparison_count + 1)  # More comparisons add stronger influence

        # ✅ Apply logarithmic likelihood with weight
        likelihood += weight * np.log(p1 if winner == img1 else p2)

    return -likelihood


# ✅ Fetch Rankings from Firestore and Apply Bradley-Terry Model
stored_comparisons = fetch_all_comparisons(school_name, year_group)

if stored_comparisons:
    rankings = calculate_rankings(stored_comparisons)

    # ✅ Store Rankings in Firestore with comparison count
    for image, score in rankings.items():
        doc_id = hashlib.sha256(image.encode()).hexdigest()[:20]  # Create a short, unique ID

        db.collection("rankings").document(doc_id).set({
            "school": school_name,
            "year_group": year_group,
            "image_url": image,
            "score": float(score),  # ✅ Store as float for consistency
            "comparison_count": stored_comparisons.count(image)  # ✅ Store number of times this image was compared
        }, merge=True)


# ✅ Define function to fetch ranked images before calling it
def fetch_ranked_images(school_name, year_group):
    """Fetches all ranked images from Firestore and sorts them by score."""
    try:
        docs = (
            db.collection("rankings")
            .where("school", "==", school_name)
            .where("year_group", "==", year_group)
            .order_by("score", direction=firestore.Query.DESCENDING)  # ✅ Sort in Firestore query
            .stream()  # ✅ Now properly aligned
        )

        scores = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("year_group") == year_group:
                scores.append((data["image_url"], data.get("score", 0), data.get("comparison_count", 0)))  # ✅ Include count

        return scores  # ✅ Already sorted in Firestore, no need to re-sort in Python

    except Exception as e:
        st.error(f"❌ Failed to fetch ranked images: {str(e)}")
        return []



# ✅ Now call `fetch_ranked_images` at the correct location
ranked_images = fetch_ranked_images(school_name, year_group)

# ✅ Display Rankings in a Table
if ranked_images:
    df = pd.DataFrame(ranked_images, columns=["Writing Sample", "Score", "Comparison Count"])

    # ✅ Apply GDS, EXS, WTS thresholds based on percentiles
    wts_cutoff = np.percentile(df["Score"], 25)
    gds_cutoff = np.percentile(df["Score"], 75)
    df["Standard"] = df["Score"].apply(lambda x: "GDS" if x >= gds_cutoff else ("WTS" if x <= wts_cutoff else "EXS"))

    st.subheader("Ranked Writing Samples")
    st.dataframe(df)
    st.sidebar.download_button("Download Results as CSV", df.to_csv(index=False).encode("utf-8"), "writing_rankings.csv", "text/csv")
else:
    st.warning("⚠️ No ranked images found for this year group. Begin voting to generate rankings.")

