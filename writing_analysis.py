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

# === STREAMLIT PAGE SETUP === #
st.set_page_config(layout="wide")
st.title("Comparative Judgement Writing Assessment")

# ✅ Initialize Firebase using `st.secrets`
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

    FIREBASE_CREDENTIALS_PATH = "/tmp/firebase_credentials.json"
    with open(FIREBASE_CREDENTIALS_PATH, "w") as json_file:
        json.dump(firebase_config, json_file)

    # ✅ Initialize Firebase
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {"storageBucket": "writing-comparison.firebasestorage.app"})

# ✅ Initialize Firestore and Storage Client
db = firestore.client()
bucket = storage.bucket()

# ✅ Debugging: Check if Firebase is initialized
if st.session_state.get("debug_mode", False):
    st.write("DEBUG: Firebase initialized successfully")



# ✅ Define fetch_all_comparisons
def fetch_all_comparisons(school_name, year_group):
    """Fetches all writing comparisons for a given school and year group from Firestore."""
    db = firestore.client()
    comparisons_ref = db.collection("comparisons")\
                        .where("school", "==", school_name)\
                        .where("year_group", "==", year_group)\
                        .stream()
    
    return [doc.to_dict() for doc in comparisons_ref]

# 🔹 Call the function where needed
if "year_group" in st.session_state and st.session_state.year_group:
    docs = db.collection("writing_samples")\
            .where(filter=firestore.FieldFilter("school", "==", school_name))\
            .where(filter=firestore.FieldFilter("year_group", "==", year_group))\
            .stream()

else:
    docs = []
    st.warning("⚠️ Please select a year group first.")


# ✅ Debug Mode Toggle (set to False for normal use, True for debugging)
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# === SCHOOL LOGINS === #
SCHOOL_CREDENTIALS = {
    "School_A": hashlib.sha256("passwordA".encode()).hexdigest(),
    "School_B": hashlib.sha256("passwordB".encode()).hexdigest(),
    "adminkbrown": hashlib.sha256("115413Gtcs@".encode()).hexdigest()
}

# === SESSION STATE INITIALIZATION === #
st.session_state.setdefault("debug_mode", False)  # ✅ Ensure debug mode is set
st.session_state.setdefault("firebase_initialized", False)  # ✅ Prevent duplicate Firebase initialization
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("school_name", "")
st.session_state.setdefault("year_group", "Year 1")  # ✅ Default to a properly formatted year group
st.session_state.setdefault("image_urls", [])
st.session_state.setdefault("pairings", [])
st.session_state.setdefault("comparisons", [])
st.session_state.setdefault("rankings", [])
st.session_state.setdefault("image_comparison_counts", {})

# ✅ Ensure year_group formatting is consistent
if "year_group" in st.session_state and st.session_state.year_group:
    clean_year_group = st.session_state.year_group.replace("Year ", "").strip()
    st.session_state.year_group = f"Year {clean_year_group}"


year_group = st.session_state.get("year_group", "Year 1")  # ✅ Use default "Year 1" if missing


# ✅ Ensure login form only appears if user is not logged in
if not st.session_state.logged_in:
    with st.sidebar:
        st.header("Login")
        school_name = st.text_input("Enter School Name", key="school_input").strip()
        password = st.text_input("Enter Password", type="password", help="Case-sensitive", key="password_input")

        login_button = st.button("Login")

    # ✅ Prevent blank submissions
    if login_button:
        if not school_name or not password:
            st.sidebar.warning("Please enter both school name and password.")
        elif school_name in SCHOOL_CREDENTIALS and hashlib.sha256(password.encode()).hexdigest() == SCHOOL_CREDENTIALS[school_name]:
            st.session_state.logged_in = True
            st.session_state.school_name = school_name
            st.sidebar.success(f"Logged in as {school_name}")
            st.rerun()  # ✅ Ensure UI updates immediately
        else:
            st.sidebar.error("Invalid credentials. Please check your username and password.")

else:
    with st.sidebar:
        st.header(f"Logged in as {st.session_state.school_name}")
        logout_button = st.button("Logout")

    if logout_button:
        keys_to_clear = ["logged_in", "school_name", "year_group"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.sidebar.info("You have been logged out.")  # ✅ Provide UI feedback
        st.rerun()

# ✅ Display Comparisons for User Selection
st.subheader("Choose the best writing sample from below:")

if "pairings" in st.session_state and st.session_state.pairings:
    for i, (img1, img2) in enumerate(st.session_state.pairings[:10]):  # Display first 10 pairs
        st.write(f"### Comparison {i + 1}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, caption="Writing Sample 1", use_column_width=True)
            if st.button(f"Select Image 1 - {i}", key=f"btn1_{i}"):
                store_comparison(img1, img2, st.session_state.school_name, st.session_state.year_group, img1)
                st.success(f"✅ You selected Image 1")

        with col2:
            st.image(img2, caption="Writing Sample 2", use_column_width=True)
            if st.button(f"Select Image 2 - {i}", key=f"btn2_{i}"):
                store_comparison(img1, img2, st.session_state.school_name, st.session_state.year_group, img2)
                st.success(f"✅ You selected Image 2")

else:
    st.warning("⚠️ No image pairs available. Ensure images are uploaded and paired first.")



def store_comparison(img1, img2, school_name, year_group, winner):
    """Stores the user's comparison selection in Firestore and ensures data integrity."""
    try:
        # ✅ Generate a unique ID for this comparison
        comparison_id = f"{school_name}_{year_group}_{hashlib.sha256((img1 + img2).encode()).hexdigest()[:20]}"
        comparison_ref = db.collection("comparisons").document(comparison_id)

        # ✅ Debug: Check if the document exists
        comparison_doc = comparison_ref.get()
        st.write(f"DEBUG: Checking if comparison {comparison_id} exists...")

        if not comparison_doc.exists:
            # ✅ Create new document if missing
            comparison_ref.set({
                "school": school_name,
                "year_group": year_group,
                "image_1": img1,
                "image_2": img2,
                "winner": winner,
                "comparison_count": 1,  # Initialize counter
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            st.success(f"✅ New comparison stored for {img1} vs {img2}")
            st.write(f"DEBUG: New comparison created: {comparison_id}")

        else:
            # ✅ Increment comparison count if document already exists
            comparison_ref.update({
                "comparison_count": firestore.Increment(1),
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            st.success(f"✅ Comparison updated for {img1} vs {img2}")
            st.write(f"DEBUG: Existing comparison updated: {comparison_id}")

    except Exception as e:
        st.error(f"❌ Failed to store comparison: {str(e)}")
        st.write(f"DEBUG: Error storing comparison - {e}")


# ✅ Define function to store user ranking data (Moved to Global Scope)
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
            K = 1.0  
            expected_score = 1 / (1 + np.exp(other_score - selected_score))  

            selected_score += K * (1 - expected_score)  
            other_score -= K * expected_score  

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

# === AFTER LOGIN === #
if st.session_state.logged_in:
    school_name = st.session_state.school_name
    st.sidebar.header("Select Year Group")
    year_group = st.sidebar.selectbox("Select Year Group", ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"])

    # ✅ Ensure the selected year group updates session state correctly
    if year_group != st.session_state.get("year_group", ""):
        st.session_state.year_group = year_group
        st.session_state.image_urls = []
        st.session_state.image_comparison_counts = {}

    # ✅ Upload Section
    st.sidebar.header("Upload Writing Samples")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Writing Samples", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key=year_group
    )

    grade_labels = {}

    if uploaded_files:
        with st.sidebar.form("upload_form"):
            for uploaded_file in uploaded_files:
                grade_labels[uploaded_file.name] = st.selectbox(
                    f"Label for {uploaded_file.name}", ["GDS", "EXS", "WTS"]
                )

            submit_button = st.form_submit_button("Confirm Upload")

        if submit_button:
            uploaded_image_urls = []
            existing_urls = set(st.session_state.image_urls)

            for uploaded_file in uploaded_files:
                grade_label = grade_labels.get(uploaded_file.name, "EXS")
                filename = f"{school_name}_{year_group}_{grade_label}_{hashlib.sha256(uploaded_file.name.encode()).hexdigest()[:10]}.jpg"
                firebase_path = f"writing_samples/{school_name}/{year_group}/{grade_label}/{filename}"
                image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{firebase_path.replace('/', '%2F')}?alt=media"

                if image_url in existing_urls:
                    st.sidebar.warning(f"⚠️ {uploaded_file.name} was already uploaded. Skipping.")
                    continue

                try:
                    # ✅ Upload file to Firebase Storage
                    blob = bucket.blob(firebase_path)
                    blob.upload_from_file(uploaded_file, content_type="image/jpeg")

                    # ✅ Store metadata in Firestore
                    db.collection("writing_samples").add({
                        "school": school_name,
                        "year_group": year_group,
                        "image_url": image_url,
                        "filename": filename,
                        "grade_label": grade_label
                    })

                    uploaded_image_urls.append(image_url)
                    st.sidebar.success(f"{uploaded_file.name} uploaded successfully as {grade_label}")

                    # ✅ Debug: Print uploaded file info
                    st.write(f"DEBUG: Uploaded Image {uploaded_file.name} → URL: {image_url}")

                except Exception as e:
                    st.sidebar.error(f"❌ Upload Failed: {str(e)}")

            if uploaded_image_urls:
                st.session_state.image_urls.extend(uploaded_image_urls)

            # ✅ Debugging: Ensure uploaded images are stored
            st.write("DEBUG: Uploaded Images", st.session_state.image_urls)

# ✅ Ensure Firebase is initialized BEFORE fetching images
if not firebase_admin._apps:
    st.error("❌ Firebase is not initialized. Please check your configuration.")
    st.stop()

# ✅ Fetch images from Firestore
docs = db.collection("writing_samples")\
            .where(filter=firestore.FieldFilter("school", "==", school_name))\
            .where(filter=firestore.FieldFilter("year_group", "==", year_group))\
            .stream()

doc_list = list(docs)  # Convert generator to list

if not doc_list:
    st.warning("⚠️ No documents retrieved from Firestore. Please check your query.")
    st.stop()

st.write("DEBUG: Retrieved Documents", [doc.to_dict() for doc in doc_list])

retrieved_images = []
for doc in doc_list:
    data = doc.to_dict()
    if "image_url" in data:
        retrieved_images.append(data["image_url"])
        st.write("DEBUG: Image Retrieved", data["image_url"])

# ✅ Ensure images are retrieved successfully
if not retrieved_images:
    st.warning("⚠️ No images found in Firestore for this year group. Please upload images.")
    st.stop()

# ✅ Store retrieved images in session state
st.session_state.image_urls = retrieved_images
st.write("DEBUG: Final Retrieved Images", st.session_state.image_urls)  # ✅ Debugging retrieval

# ✅ Ensure `image_pool` is initialized
image_pool = {"GDS": [], "EXS": [], "WTS": []}

# ✅ Display images properly before generating pairings
if st.session_state.image_urls:
    st.image(
        st.session_state.image_urls,
        caption=["Uploaded Writing Samples"] * len(st.session_state.image_urls),  # Fix caption length
        use_container_width=True
    )
else:
    st.warning("⚠️ No images available for this year group.")


# ✅ Debug: Ensure documents are retrieved
st.write("DEBUG: Retrieved Documents", [doc.to_dict() for doc in doc_list])

# ✅ Populate `image_pool` with retrieved images
for doc in doc_list:
    data = doc.to_dict()
    if "image_url" in data and "grade_label" in data:
        if data["grade_label"] in image_pool:
            image_pool[data["grade_label"]].append(data["image_url"])
        else:
            st.warning(f"⚠️ Image {data['image_url']} has an unknown grade label and won't be paired.")

# ✅ Ensure `sample_pool` is initialized
sample_pool = {"GDS": [], "EXS": [], "WTS": []}

# ✅ Populate `sample_pool` with images
for img_url in st.session_state.image_urls:
    for grade in image_pool.keys():
        if img_url in image_pool[grade]:
            sample_pool[grade].append(img_url)

st.write("DEBUG: Sample Pool Populated", sample_pool)

# ✅ Define a maximum number of pairs based on available images
max_pairs = min(40, sum(len(sample_pool[k]) for k in sample_pool) // 2)

all_pairs = set()
pairing_attempts = {"GDS": 0, "EXS": 0, "WTS": 0}

# ✅ Ensure there are enough images before entering the loop
if sum(len(sample_pool[k]) for k in sample_pool) < 2:
    st.warning("⚠️ Not enough images available to generate pairs.")
    st.stop()

# ✅ Adaptive Pairing Logic
def select_pair(sample_pool, selected_grade):
    """ Selects the most appropriate pair while ensuring fair comparisons across categories. """
    if selected_grade == "GDS" and sample_pool["EXS"]:
        return (random.choice(sample_pool["GDS"]), random.choice(sample_pool["EXS"]))  # GDS vs EXS
    elif selected_grade == "GDS" and sample_pool["WTS"]:
        return (random.choice(sample_pool["GDS"]), random.choice(sample_pool["WTS"]))  # GDS vs WTS
    elif selected_grade == "EXS" and sample_pool["WTS"]:
        return (random.choice(sample_pool["EXS"]), random.choice(sample_pool["WTS"]))  # EXS vs WTS
    elif selected_grade == "EXS" and sample_pool["GDS"]:
        return (random.choice(sample_pool["EXS"]), random.choice(sample_pool["GDS"]))  # EXS vs GDS
    elif selected_grade == "WTS" and sample_pool["EXS"]:
        return (random.choice(sample_pool["WTS"]), random.choice(sample_pool["EXS"]))  # WTS vs EXS
    elif selected_grade == "WTS" and sample_pool["GDS"]:
        return (random.choice(sample_pool["WTS"]), random.choice(sample_pool["GDS"]))  # WTS vs GDS
    elif len(sample_pool[selected_grade]) > 1:
        return tuple(random.sample(sample_pool[selected_grade], 2))  # Fallback: Same grade pairing
    return None

# ✅ Generate pairs ensuring fair comparisons
while len(all_pairs) < max_pairs:
    # Select a grade based on remaining images
    selected_grade = random.choices(
        list(sample_pool.keys()), 
        weights=[len(sample_pool[g]) for g in sample_pool if sample_pool[g]]
    )[0]

    pair = select_pair(sample_pool, selected_grade)

    # ✅ Ensure unique pairings
    if pair and pair not in all_pairs and (pair[1], pair[0]) not in all_pairs:
        all_pairs.add(pair)
        pairing_attempts[selected_grade] += 1

st.write("DEBUG: Generated Pairs", list(all_pairs))

# ✅ Store the selected pairs
st.session_state.pairings = list(all_pairs)


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
if "year_group" in st.session_state and st.session_state.year_group:
    stored_comparisons = fetch_all_comparisons(st.session_state.school_name, st.session_state.year_group)
else:
    stored_comparisons = []
    st.warning("⚠️ Please select a year group first.")

# ✅ Ensure rankings are only calculated if there are comparisons
if stored_comparisons:
    rankings = calculate_rankings(stored_comparisons)

    # ✅ Store Rankings in Firestore with comparison count
    for image, score in rankings.items():
        doc_id = hashlib.sha256(image.encode()).hexdigest()[:20]  # Create a short, unique ID

        db.collection("rankings").document(doc_id).set({
            "school": st.session_state.school_name,  # ✅ Ensure session state is used
            "year_group": st.session_state.year_group,  # ✅ Ensure session state is used
            "image_url": image,
            "score": float(score),  # ✅ Store as float for consistency
            "comparison_count": sum(1 for comp in stored_comparisons if image in comp)  # ✅ Correct count logic
        }, merge=True)
else:
    st.warning("⚠️ No valid comparisons found for ranking. Make more comparisons first.")

# ✅ Define function to fetch ranked images before calling it
def fetch_ranked_images(school_name, year_group):
    """Fetches all ranked images from Firestore and sorts them by score."""
    try:
        # ✅ Ensure year_group is formatted correctly before querying Firestore
        if not year_group.startswith("Year "):
            clean_year_group = year_group.replace("Year ", "").strip()
            year_group = f"Year {clean_year_group}"

                # ✅ Fetch documents from Firestore (filter out missing scores)
        docs = (
            db.collection("writing_samples")
              .where(filter=firestore.FieldFilter("school", "==", school_name))
              .where(filter=firestore.FieldFilter("year_group", "==", year_group))
              .where(filter=firestore.FieldFilter("score", ">=", 0))  # ✅ Ensure only scored images are retrieved
              .order_by("score", direction=firestore.Query.DESCENDING)  # ✅ Sort by score
              .stream()
        )

        doc_list = list(docs)  # Convert generator to list

        # ✅ Debug: Check if docs are retrieved
        if not doc_list:
            st.warning("⚠️ No ranked images found. Make more comparisons first.")
            return []
        
        st.write("DEBUG: Retrieved Ranked Documents", [doc.to_dict() for doc in doc_list])

        scores = []
        for doc in doc_list:
            data = doc.to_dict()
            image_url = data.get("image_url")
            score = data.get("score", 0)
            comparison_count = data.get("comparison_count", 0)

            # ✅ Validate and append only if image URL exists
            if image_url:
                scores.append((image_url, score, comparison_count))
            else:
                st.warning(f"⚠️ Document missing image_url: {data}")

        return scores  # ✅ Sorted by score

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

