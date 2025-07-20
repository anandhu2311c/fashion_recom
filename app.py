import streamlit as st
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize
import matplotlib.image as mpimg
from PIL import Image

# ------------------- Load Models -------------------
encoder = tf.keras.models.load_model("encoder.keras", compile=False)
pm_model = tf.keras.models.load_model("pm_model.keras", compile=False)

# ------------------- Load Dataset -------------------
df = pd.read_csv("real_fashion_data.csv")
df["rating"] = pd.to_numeric(df["rating"], errors='coerce').fillna(0.0)

# ------------------- Encode Images -------------------
image_encodings = []
valid_indices = []

for i in range(len(df)):
    img_path = f"cropped_tshirts/tshirt_{i}_0.jpg"
    if os.path.exists(img_path):
        try:
            img = mpimg.imread(img_path)
            img = resize(img, (320, 192, 3))
            enc = encoder.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            image_encodings.append(enc.flatten())
            valid_indices.append(i)
        except Exception as e:
            print(f"❌ Error reading {img_path}: {e}")

# Filter and update DataFrame
df = df.loc[valid_indices].reset_index(drop=True)
df["encodings"] = image_encodings

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Fashion Recommendation Engine", layout="wide")
st.title("👕 AI-Powered Fashion Recommender")

uploaded_file = st.file_uploader("Upload a T-shirt image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_file).convert('RGB')
    img_resized = resize(np.array(img), (320, 192, 3))

    # Encode and predict popularity
    encoding = encoder.predict(np.expand_dims(img_resized, axis=0), verbose=0)[0].flatten()
    popularity = pm_model.predict(np.array([encoding]), verbose=0)[0][0]

    st.markdown(f"🎯 **Predicted Popularity Score:** `{popularity:.2f}`")

    # Recommend similar
    all_encodings = np.vstack(df["encodings"].values)
    similarities = cosine_similarity([encoding], all_encodings)[0]
    top_indices = similarities.argsort()[::-1][:5]

    st.subheader("🔍 Recommended Similar Products")

    cols = st.columns(5)
    for idx, col in zip(top_indices, cols):
        row = df.iloc[idx]
        image_path = f"cropped_tshirts/tshirt_{idx}_0.jpg"
        if os.path.exists(image_path):
            col.image(image_path, use_column_width=True)
            col.markdown(f"⭐ **Rating**: `{row['rating']:.1f}`")
            col.markdown(f"🔁 **Similarity**: `{similarities[idx]:.2f}`")
            if pd.notna(row.get("product_url", "")):
                col.markdown(f"[🔗 View Product]({row['product_url']})", unsafe_allow_html=True)
