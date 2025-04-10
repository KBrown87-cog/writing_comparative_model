import numpy as np
import streamlit as st
import hashlib
from scipy.optimize import minimize
from google.cloud.firestore_v1 import FieldFilter
from firebase_admin import firestore

@st.cache_data(ttl=60)
def fetch_all_comparisons(school_name, year_group):
    """Fetches all writing comparisons for a given school and year group from Firestore."""
    db = st.session_state["firestore_client"]  # ✅ ADD THIS

    if not school_name or not year_group:
        raise ValueError("❌ Invalid school name or year group provided.")

    try:
        comparisons_ref = db.collection("comparisons")\
            .where(filter=FieldFilter("school", "==", school_name))\
            .where(filter=FieldFilter("year_group", "==", year_group))\
            .stream()

        comparisons = [doc.to_dict() for doc in comparisons_ref]

        if not comparisons:
            raise ValueError("⚠️ No comparisons found for this school and year group.")

        return comparisons

    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch comparisons from Firestore: {str(e)}")

def bradley_terry_log_likelihood(scores, comparisons, comparison_counts):
    """Calculates likelihood for Bradley-Terry ranking with weighting and numerical stability."""
    likelihood = 0

    if not comparisons:
        st.warning("⚠️ No comparisons available for ranking.")
        return None  # ✅ Prevents invalid calculations

    for comparison in comparisons:
        img1 = comparison.get("image_1")
        img2 = comparison.get("image_2")
        winner = comparison.get("winner")

        if not all([img1, img2, winner]):
            continue  # Skip invalid comparisons

        s1, s2 = scores.get(img1, 0), scores.get(img2, 0)  # Default scores

        # ✅ Log-Sum-Exp Trick for Numerical Stability
        max_score = max(s1, s2)
        exp_s1, exp_s2 = np.exp(s1 - max_score), np.exp(s2 - max_score)
        p1 = exp_s1 / (exp_s1 + exp_s2)
        p2 = exp_s2 / (exp_s1 + exp_s2)

        # ✅ Smoothed Weighting Formula
        weight = np.log1p(comparison_counts.get(img1, 1))  

        # ✅ Apply logarithmic likelihood with weight
        likelihood += weight * np.log(p1 if winner == img1 else p2)

    return -likelihood  # ✅ Negative log-likelihood for minimization


def calculate_rankings(comparisons):
    """Applies Bradley-Terry Model to rank images, incorporating weighting and convergence checks."""
    
    if not comparisons:
        st.warning("⚠️ No valid comparisons available. Ranking cannot be calculated yet.")
        return None

    comparison_counts = {}
    valid_comparisons = []

    for comparison in comparisons:
        img1 = comparison.get("image_1")
        img2 = comparison.get("image_2")

        winner = comparison.get("winner")
        if not winner:
            winners_list = comparison.get("winners", [])
            if winners_list:
                winner = max(set(winners_list), key=winners_list.count)

        comparison_count = comparison.get("comparison_count", 1)

        if not all([img1, img2, winner]):
            st.warning(f"⚠️ Invalid comparison data found: {comparison}")
            continue

        comparison_counts[img1] = comparison_counts.get(img1, 0) + comparison_count
        comparison_counts[img2] = comparison_counts.get(img2, 0) + comparison_count

        valid_comparisons.append({
            "image_1": img1,
            "image_2": img2,
            "winner": winner
        })

    sample_names = [name for name in comparison_counts.keys() if comparison_counts[name] > 0]

    if not sample_names:
        st.warning("⚠️ No valid image comparisons available for ranking.")
        return None

    initial_scores = {name: np.random.uniform(-0.1, 0.1) for name in sample_names}

    try:
        result = minimize(
            lambda s: bradley_terry_log_likelihood(
                dict(zip(sample_names, s)),
                valid_comparisons,
                comparison_counts
            ),
            list(initial_scores.values()), 
            method='L-BFGS-B'
        )

        if not result.success:
            st.warning("⚠️ Optimization failed to converge. Using default scores.")
            return {name: 50 for name in sample_names}

        raw_scores = dict(zip(sample_names, result.x))
        min_score, max_score = min(raw_scores.values()), max(raw_scores.values())

        if max_score == min_score:
            return {name: 50 for name in sample_names}

        normalized_scores = {
            name: 100 * (score - min_score) / (max_score - min_score)
            for name, score in raw_scores.items()
        }

        # ✅ NEW: Store comparison counts for use when saving
        st.session_state["comparison_counts"] = comparison_counts

        return normalized_scores

    except Exception as e:
        st.error(f"❌ Ranking Calculation Failed: {str(e)}")
        return None



def save_rankings_to_firestore(rankings, school_name, year_group):
    """Saves the normalized ranking scores into Firestore under the 'rankings' collection."""
    db = st.session_state["firestore_client"]

    try:
        # ✅ Fetch grade labels from Firestore instead of relying on session state
        docs = (
            db.collection("writing_samples")
              .where(filter=FieldFilter("school", "==", school_name))
              .where(filter=FieldFilter("year_group", "==", year_group))
              .stream()
        )

        label_lookup = {
            doc.to_dict()["image_url"]: doc.to_dict().get("grade_label", "Not Provided")
            for doc in docs if "image_url" in doc.to_dict()
        }

        for image_url, score in rankings.items():
            doc_id = hashlib.sha256(image_url.encode()).hexdigest()
            doc_ref = db.collection("rankings").document(doc_id)

            doc_ref.set({
                "school": school_name,
                "year_group": year_group,
                "image_url": image_url,
                "score": score,
                "comparison_count": st.session_state.comparison_counts.get(image_url, 0),
                "grade_label": label_lookup.get(image_url, "Not Provided"),  # ✅ Reliable source now
                "timestamp": firestore.SERVER_TIMESTAMP
            })

    except Exception as e:
        st.error(f"❌ Failed to save rankings: {str(e)}")
