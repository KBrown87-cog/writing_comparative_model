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
import io
import re
from PIL import Image
import time
from collections import defaultdict
from google.cloud.firestore_v1 import FieldFilter
from helpers import calculate_rankings, fetch_all_comparisons, save_rankings_to_firestore

# ✅ Force defaults early to prevent Streamlit Cloud bug
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# === STREAMLIT PAGE SETUP === #
st.set_page_config(layout="wide")
st.markdown("<h1 style='color: #f5f5f5; text-align: center;'>Comparative Judgement Writing Assessment</h1>", unsafe_allow_html=True)


# === ✅ Now Initialize Session State === #

# ✅ Ensure session state variables exist
st.session_state.setdefault("debug_mode", False)  
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("school_name", "")  # ✅ Ensures `school_name` exists
st.session_state.setdefault("year_group", "Year 1")
st.session_state.setdefault("image_urls", [])
st.session_state.setdefault("pairings", [])  # ✅ Ensure pairings persist across reruns
st.session_state.setdefault("comparisons", [])
st.session_state.setdefault("rankings", [])
st.session_state.setdefault("image_comparison_counts", {})
st.session_state.setdefault("failed_attempts", {})  # ✅ Ensure failed attempts tracking
st.session_state.setdefault("sample_pool", defaultdict(list))  # ✅ Store categorized writing samples
st.session_state.setdefault("comparison_counts", {})  # ✅ Track how many times each pair has been compared
st.session_state.setdefault("used_images", set())  # ✅ Track used images to prevent duplicates
st.session_state.setdefault("selection_locked", False)
st.session_state.setdefault("generated_pairs", set())
st.session_state.setdefault("samples_with_labels", [])


# === FORMAT YEAR GROUP === #
if st.session_state.get("year_group"):
    clean_year_group = re.sub(r"\D", "", st.session_state["year_group"])  
    if clean_year_group.isdigit():
        st.session_state["year_group"] = f"Year {clean_year_group}"
    else:
        st.session_state["year_group"] = "Year 1" 


# ✅ Ensure `st.secrets["CREDENTIALS"]` exists before using it
if "CREDENTIALS" in st.secrets:
    SCHOOL_CREDENTIALS = {
        school: hashlib.sha256(st.secrets["CREDENTIALS"][school].encode()).hexdigest()
        for school in st.secrets["CREDENTIALS"]
    }
else:
    st.stop()


# ✅ Initialize Firebase using `st.secrets`, ensuring it only runs once
if not firebase_admin._apps:
    try:
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

        cred_json = json.dumps(firebase_config)
        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred, {"storageBucket": st.secrets["FIREBASE"]["STORAGE_BUCKET"]})

        st.session_state["firestore_client"] = firestore.client()
        st.session_state["storage_bucket"] = storage.bucket()
        st.session_state["firebase_initialized"] = True
    except Exception as e:
        st.error(f"❌ Firebase initialization failed: {str(e)}")
        st.stop()

try:
    db = st.session_state["firestore_client"]
    bucket = st.session_state["storage_bucket"]
except KeyError:
    st.error("❌ Firestore or Storage is not initialized.")
    st.stop()


# === HELPER FUNCTIONS: FIRESTORE FETCHING === #

@st.cache_data(ttl=60)  # ✅ Cache Firestore data for 60 seconds
def fetch_images(school_name, year_group):
    """Fetch all uploaded images from Firestore for a given school and year group."""
    if not school_name or not year_group:
        raise ValueError("❌ School name or year group is missing. Cannot fetch images.")

    try:
        docs = (
            db.collection("writing_samples")
              .where("school", "==", school_name)
              .where("year_group", "==", year_group)
              .stream()
        )
        
        images = [doc.to_dict().get("image_url") for doc in docs if doc.to_dict().get("image_url")]

        if not images:
            raise ValueError("⚠️ No images found in Firestore for this school and year group.")

        return images

    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch images from Firestore: {str(e)}")


def store_comparison(img1, img2, school_name, year_group, winner):
    """Stores the user's comparison selection in Firestore and ensures data integrity."""
    try:
        if "firestore_client" not in st.session_state:
            st.error("❌ Firestore is not initialized.")
            return
        
        db = st.session_state["firestore_client"]

        if not all([img1, img2, school_name, year_group, winner]):
            st.error("❌ Invalid input: One or more required fields are missing.")
            return

        comparison_id = f"{school_name}_{year_group}_{hashlib.sha256('_'.join(sorted([img1, img2])).encode()).hexdigest()[:20]}"
        comparison_ref = db.collection("comparisons").document(comparison_id)
        comparison_doc = comparison_ref.get()

        if not comparison_doc.exists:
            comparison_ref.set({
                "school": school_name,
                "year_group": year_group,
                "image_1": img1,
                "image_2": img2,
                "winners": [winner],
                "comparison_count": 1,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        else:
            existing_data = comparison_doc.to_dict()
            previous_winners = existing_data.get("winners", [])
            previous_winners.append(winner)
            most_voted_winner = max(set(previous_winners), key=previous_winners.count)

            comparison_ref.update({
                "comparison_count": firestore.Increment(1),
                "winners": previous_winners,
                "winner": most_voted_winner,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

    except Exception as e:
        st.error(f"❌ Failed to store comparison: {str(e)}")


def select_pair(sample_pool, used_pairs, max_retries=10):
    """Selects a new unique pair from all category combinations including same-category pairs."""
    available_categories = [k for k in sample_pool if sample_pool[k]]

    if len(available_categories) == 0:
        return None

    retries = 0
    while retries < max_retries:
        retries += 1

        # ✅ Now includes intra-category via combinations_with_replacement
        for cat1, cat2 in itertools.combinations_with_replacement(available_categories, 2):
            if not sample_pool[cat1] or not sample_pool[cat2]:
                continue

            img1 = random.choice(sample_pool[cat1])
            img2 = random.choice(sample_pool[cat2])

            if img1 == img2:
                continue  # Avoid pairing image with itself

            pair_key = tuple(sorted([img1, img2]))

            if pair_key in used_pairs:
                continue

            # ✅ Mark as used
            used_pairs.add(pair_key)
            st.session_state.used_images.update([img1, img2])
            return img1, img2

    return None

def generate_pairings(sample_pool, max_retries=10):
    """Generate all possible unique image pairs from the sample pool across and within categories."""

    categories = list(sample_pool.keys())
    all_pairs = set()
    used_pairs = st.session_state.get("generated_pairs", set())

    # === Build all possible pairs: cross-band AND intra-band === #
    for cat1 in categories:
        samples1 = sample_pool[cat1]

        # Intra-band (e.g., GDS vs GDS)
        for i in range(len(samples1)):
            for j in range(i + 1, len(samples1)):
                img1, img2 = samples1[i], samples1[j]
                pair_key = tuple(sorted([img1, img2]))
                if pair_key not in used_pairs:
                    all_pairs.add(pair_key)

        # Cross-band (e.g., GDS vs EXS)
        for cat2 in categories:
            if cat1 >= cat2:  # Avoid duplicate cat combinations and self-pairing
                continue
            samples2 = sample_pool[cat2]
            for img1 in samples1:
                for img2 in samples2:
                    pair_key = tuple(sorted([img1, img2]))
                    if pair_key not in used_pairs:
                        all_pairs.add(pair_key)

    # ✅ Add new pairs up to retry limits (if enabled)
    final_pairs = []
    for pair_key in all_pairs:
        current_count = st.session_state.comparison_counts.get(pair_key, 0)
        if current_count < 5:  # ✅ Only include if not over-compared
            final_pairs.append(pair_key)
            used_pairs.add(pair_key)

    st.session_state.generated_pairs = used_pairs
    return final_pairs



# ✅ Ensure login form only appears if user is not logged in
if not st.session_state.get("logged_in", False):
    # === Set background image === #
    page_bg_img = """
    <style>
    /* Full-page background image */
    [data-testid="stAppViewContainer"] {
        background: url('https://i.imgur.com/FOuu4dM.jpg') no-repeat center center fixed;
        background-size: cover;
    }

    /* Sidebar overlay with blur effect */
    [data-testid="stSidebar"] {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 20rem;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(6px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Push the main content to the right to not overlap */
    .css-1outpf7 { 
        margin-left: 20rem; 
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    hide_nav_tabs = """
    <style>
    /* Hide Streamlit's multipage navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """
    st.markdown(hide_nav_tabs, unsafe_allow_html=True)


    # === Login form in sidebar === #
    with st.sidebar:
        st.header("Login")
        school_name = st.text_input("Enter School Name", key="school_input").strip()
        password = st.text_input("Enter Password", type="password", help="Case-sensitive", key="password_input")
        login_button = st.button("Login")

        if login_button:
            if not school_name or not password:
                st.warning("Please enter both school name and password.")
            elif "SCHOOL_CREDENTIALS" not in globals():
                st.error("❌ SCHOOL_CREDENTIALS is not available. Please check your configuration.")
            elif school_name in st.session_state["failed_attempts"] and st.session_state["failed_attempts"][school_name] >= 3:
                st.error("Too many failed attempts. Try again later.")
            elif school_name and school_name in SCHOOL_CREDENTIALS:
                st.session_state["logged_in"] = True
                st.session_state["school_name"] = school_name
                st.session_state["failed_attempts"][school_name] = 0
                st.success(f"Logged in as {school_name}")
                st.rerun()
            else:
                st.session_state["failed_attempts"][school_name] = st.session_state["failed_attempts"].get(school_name, 0) + 1
                st.error(f"Invalid credentials. Attempts: {st.session_state['failed_attempts'][school_name]}/3")

# ✅ Logged-in state remains unchanged
else:
    with st.sidebar:
        st.header(f"Logged in as {st.session_state.school_name}")
        logout_button = st.button("Logout")

    if logout_button:
        keys_to_clear = ["logged_in", "school_name", "year_group", "image_urls", "pairings"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.sidebar.info("You have been logged out.")
        st.rerun()


# ✅ Only fetch comparisons if images already exist
if st.session_state.get("logged_in") and st.session_state.get("school_name") and st.session_state.get("image_urls"):
    try:
        comparisons = fetch_all_comparisons(
            st.session_state["school_name"],
            st.session_state["year_group"]
        )
        st.session_state["comparisons"] = comparisons
    except Exception as e:
        # Don't show errors unless there are comparisons
        if "No comparisons found" not in str(e):
            st.warning(str(e))
else:
    comparisons = []

# === AFTER LOGIN === #
if st.session_state.logged_in:
    school_name = st.session_state["school_name"]

    # === YEAR GROUP SELECTION === #
    st.sidebar.header("Select Year Group")

    # Set a default year group in session state
    st.session_state.setdefault("year_group", "Year 1")

    # Render the selectbox
    selected_year = st.sidebar.selectbox(
        "Select Year Group",
        ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"],
        index=["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6"].index(st.session_state["year_group"])
    )

    # If changed, update and rerun
    if selected_year != st.session_state["year_group"]:
        st.session_state["year_group"] = selected_year
        st.session_state.image_urls = []
        st.session_state.image_comparison_counts = {}
        st.rerun()

    # ✅ Set local shortcut for easier use
    year_group = st.session_state["year_group"]

    # === IMAGE UPLOAD SECTION === #
    st.sidebar.header("Upload Writing Samples")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Writing Samples",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=year_group  # ✅ Now safely defined
    )

    # ✅ Init session storage for grading info
    st.session_state.setdefault("samples_with_labels", [])

    if uploaded_files:
        with st.sidebar.form("upload_form"):
            grade_labels = {
                file.name: st.selectbox(f"Label for {file.name}", ["GDS", "EXS", "WTS"])
                for file in uploaded_files
            }
            submit_button = st.form_submit_button("Confirm Upload")

        if submit_button:
            uploaded_image_urls = []
            existing_urls = set(st.session_state.image_urls)
            batch = db.batch()

            for uploaded_file in uploaded_files:
                grade_label = grade_labels.get(uploaded_file.name, "EXS")
                filename = f"{school_name}_{year_group}_{grade_label}_{hashlib.sha256(uploaded_file.name.encode()).hexdigest()[:10]}.jpg"
                firebase_path = f"writing_samples/{school_name}/{year_group}/{grade_label}/{filename}"
                image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{firebase_path.replace('/', '%2F')}?alt=media"

                if image_url in existing_urls:
                    st.sidebar.warning(f"⚠️ {uploaded_file.name} already uploaded. Skipping.")
                    continue

                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    image.thumbnail((1024, 1024))
                    img_io = io.BytesIO()
                    image.save(img_io, format="JPEG", quality=85)
                    img_io.seek(0)

                    blob = bucket.blob(firebase_path)
                    blob.upload_from_file(img_io, content_type="image/jpeg")

                    doc_ref = db.collection("writing_samples").document()
                    batch.set(doc_ref, {
                        "school": school_name,
                        "year_group": year_group,
                        "image_url": image_url,
                        "filename": filename,
                        "grade_label": grade_label
                    })

                    uploaded_image_urls.append(image_url)

                    # ✅ Save image + teacher label
                    st.session_state["samples_with_labels"].append({
                        "image_url": image_url,
                        "grade_label": grade_label
                    })

                    st.sidebar.success(f"{uploaded_file.name} uploaded as {grade_label}")

                except Exception as e:
                    st.sidebar.error(f"❌ Upload Failed: {str(e)}")

            batch.commit()

            for url in uploaded_image_urls:
                if url not in st.session_state.image_urls:
                    st.session_state.image_urls.append(url)


                    
    # === RANKINGS PAGE BUTTON === #
    if st.sidebar.button("📊 View Rankings"):
        st.switch_page("pages/Rankings.py")  # Path must match your `pages/` folder file name

    # === IMAGE RETRIEVAL FALLBACK === #
    if not st.session_state.image_urls:
        try:
            docs = db.collection("writing_samples")\
                .where(filter=FieldFilter("school", "==", school_name))\
                .where(filter=FieldFilter("year_group", "==", year_group))\
                .stream()
            fetched_urls = [doc.to_dict()["image_url"] for doc in docs if "image_url" in doc.to_dict()]
            st.session_state.image_urls = fetched_urls
        except Exception as e:
            st.error(f"❌ Failed to load images: {str(e)}")

    # === POPULATE SAMPLE POOL === #
    sample_pool = defaultdict(list)
    unique_image_urls = list(set(st.session_state.image_urls))

    for image_url in unique_image_urls:
        if "GDS" in image_url:
            sample_pool["GDS"].append(image_url)
        elif "EXS" in image_url:
            sample_pool["EXS"].append(image_url)
        elif "WTS" in image_url:
            sample_pool["WTS"].append(image_url)

    st.session_state["sample_pool"] = sample_pool

    # ✅ Ensure generated_pairs exists
    st.session_state.setdefault("generated_pairs", set())

    # ✅ Reset generator if empty
    if not st.session_state.get("pairings"):
        st.session_state.generated_pairs.clear()

    # === GENERATE PAIRINGS === #
    used_pairs = st.session_state.get("generated_pairs", set())
    pairings = generate_pairings(st.session_state["sample_pool"])

    if pairings:
        st.session_state.pairings = pairings

def handle_selection(winning_img, img1, img2):
    """Handles image selection, updates Firestore, and advances pairs."""
    if st.session_state.selection_locked:
        return  # ✅ Prevent multiple selections

    st.session_state.selection_locked = True  # ✅ Lock selection
    store_comparison(img1, img2, st.session_state.school_name, st.session_state.year_group, winning_img)

    st.session_state.used_images.update([img1, img2])

    # ✅ Remove the selected pair
    if st.session_state.pairings:
        st.session_state.pairings.pop(0)

    st.session_state.selection_locked = False  # ✅ Unlock for next round
    st.rerun()

# ✅ Ensure valid pair selection
if st.session_state.pairings:
    img1, img2 = st.session_state.pairings[0]

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Writing Sample 1", use_container_width=True)
        if st.button("Select Sample 1", key="img1", help="Click to choose this sample", disabled=st.session_state.selection_locked):
            handle_selection(winning_img=img1, img1=img1, img2=img2)

    with col2:
        st.image(img2, caption="Writing Sample 2", use_container_width=True)
        if st.button("Select Sample 2", key="img2", help="Click to choose this sample", disabled=st.session_state.selection_locked):
            handle_selection(winning_img=img2, img1=img1, img2=img2)

# === 📊 Calculate and Save Rankings === #
if st.session_state.get("logged_in"):
    comparisons = st.session_state.get("comparisons", [])

    if comparisons:
        rankings = calculate_rankings(comparisons)

        if rankings:
            save_rankings_to_firestore(rankings, st.session_state["school_name"], st.session_state["year_group"])
            st.session_state["rankings"] = rankings

