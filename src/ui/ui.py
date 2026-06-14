"""Streamlit UI for the Ensemble Embedding Viewer.

This lightweight frontend queries the Ensemble API to compare how multiple
embedding models interpret a user-supplied query against a chosen category
dataset. Run with `streamlit run src/ui/ui.py`. Adjust `API_URL` to point to
the ensemble service if it's not running on localhost:8000.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Ensemble Embedding Viewer", page_icon="🧠", layout="wide")

st.title("🧠 Ensemble Embedding Viewer")
st.markdown("Compare how different embedding models interpret text")

API_URL = "http://localhost:8000"

# `API_URL` should point at the running Ensemble API. Change it if your
# ensemble service is reachable at a different host/port (e.g., in Docker).

# ==========================================
# SIDEBAR: DYNAMIC DATASET MANAGEMENT
# ==========================================
with st.sidebar:
    st.header("⚙️ Manage Datasets")
    
    # Fetch the list of available datasets from the Ensemble API. If the API
    # is unreachable we present an error so the user knows to check port
    # forwarding / service connectivity.
    try:
        datasets_response = requests.get(f"{API_URL}/datasets", timeout=5)
        available_datasets = datasets_response.json().get("available_datasets", [])
    except:
        available_datasets = []

    if not available_datasets:
        st.error("🚨 Could not load datasets. Is the K8s port-forward running?")
        st.stop()

    selected_dataset = st.selectbox("Select a Dataset:", available_datasets)

    # When a dataset is selected, request its category list from the API so
    # the user can inspect, add, and remove categories dynamically.
    if selected_dataset:
        try:
            cat_response = requests.get(f"{API_URL}/datasets/{selected_dataset}", timeout=5)
            current_categories = cat_response.json().get("categories", [])
        except:
            current_categories = []

        st.write(f"**Current Categories ({len(current_categories)}):**")
        st.write(", ".join(current_categories) if current_categories else "No categories found.")

        st.divider()

        # --- ADD CATEGORY ---
        st.subheader("➕ Add Category")
        new_cat = st.text_input("New category name:", key="add_cat")
        if st.button("Add Category"):
            # Call the Ensemble API to update the dataset with the new
            # category. We replace the full category list, which keeps the
            # client logic simple and avoids partial-update races.
            if new_cat and new_cat not in current_categories:
                updated_list = current_categories + [new_cat]
                requests.post(f"{API_URL}/datasets/{selected_dataset}", json={"categories": updated_list}, timeout=5)
                st.success(f"Added '{new_cat}'!")
                st.rerun() # Force UI to refresh
            elif new_cat in current_categories:
                st.warning("Category already exists!")

        # --- REMOVE CATEGORY ---
        st.subheader("➖ Remove Category")
        if current_categories:
            cat_to_remove = st.selectbox("Select category to remove:", current_categories, key="rem_cat")
            if st.button("Remove Category"):
                # Removing is implemented by sending the filtered category
                # list back to the Ensemble API and persisting it server-side.
                updated_list = [c for c in current_categories if c != cat_to_remove]
                requests.post(f"{API_URL}/datasets/{selected_dataset}", json={"categories": updated_list}, timeout=5)
                st.success(f"Removed '{cat_to_remove}'!")
                st.rerun() # Force UI to refresh

# ==========================================
# MAIN PAGE: MODEL COMPARISON
# ==========================================
st.divider()

query = st.text_input("Enter your search query:", "Which brand is known for environmental sustainability?")

if st.button("Compare Models", type="primary", use_container_width=True):
    # Validate inputs before making the network request.
    if not query or not selected_dataset:
        st.warning("Please enter a query and select a dataset.")
    else:
        with st.spinner(f"Querying models using the '{selected_dataset}' dataset..."):
            try:
                # Ask the Ensemble API to run the query against all model
                # services and return a compact summary per model.
                response = requests.post(
                    f"{API_URL}/compare-all-db",
                    json={"dataset_name": selected_dataset, "query": query},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    st.error(f"API Error: {data['error']}")
                else:
                    st.subheader(f"Results for: *'{data['query']}'* in **{data['dataset_used']}**")
                    
                    model_names = list(data["ensemble_comparison"].keys())
                    cols = st.columns(len(model_names))
                    
                    for idx, model_name in enumerate(model_names):
                        with cols[idx]:
                            result = data["ensemble_comparison"][model_name]
                            st.markdown(f"### {model_name}")
                            
                            if "error" in result:
                                st.error(f"**Error:** {result['error']}")
                            else:
                                # Display the top category, a progress bar for
                                # the score, and a short vector preview for
                                # debugging/insight.
                                st.metric(label="Top Match", value=result["top_category"])
                                st.progress(min(result["score"] / 1.0, 1.0))
                                st.write(f"**Cosine Score:** `{result['score']:.4f}`")
                                
                                with st.expander("🧮 View Raw Vectors"):
                                    vector_data = result.get("vector_preview", [])
                                    vector_size = result.get("vector_size", "Unknown")
                                    
                                    if vector_data:
                                        formatted_vector = [round(v, 4) for v in vector_data]
                                        st.write(f"**Category Vector (first 8 dims out of {vector_size} dims):**")
                                        st.code(str(formatted_vector), language="python")
                                    else:
                                        st.write("Vector data unavailable.")
                            
            except requests.exceptions.RequestException as e:
                # Surface network errors with a helpful message to the user.
                st.error("🚨 **Could not connect to the Ensemble API.** Make sure your K8s port-forward is running!")