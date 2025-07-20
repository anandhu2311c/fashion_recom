import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize
import matplotlib.image as mpimg

# ------------------- Load Encoder & Popularity Model -------------------
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

df = df.loc[valid_indices].reset_index(drop=True)
df["encodings"] = image_encodings

print(f"✅ Loaded and encoded {len(df)} valid fashion products.")

# ------------------- Recommend + Popularity Function -------------------
def recommend_similar(image_path, top_k=5):
    if not os.path.exists(image_path):
        print("❌ Image not found.")
        return None

    try:
        img = mpimg.imread(image_path)
        img = resize(img, (320, 192, 3))
        enc = encoder.predict(np.expand_dims(img, axis=0), verbose=0)[0].flatten()

        # Predict popularity
        popularity = float(pm_model.predict(np.array([enc]), verbose=0)[0][0])

        # Compute cosine similarity
        all_encodings = np.vstack(df["encodings"].values)
        similarities = cosine_similarity([enc], all_encodings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        # Prepare results
        recommendations = df.iloc[top_indices][["name", "rating"]].copy()
        recommendations["similarity"] = similarities[top_indices]
        recommendations["popularity"] = [float(pm_model.predict(np.array([df.iloc[i]["encodings"]]), verbose=0)[0][0]) for i in top_indices]

        return popularity, recommendations

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ------------------- Run Interactive Loop -------------------
while True:
    path = input("📥 Enter image path (or type 'exit'): ").strip()
    if path.lower() == "exit":
        break

    result = recommend_similar(path, top_k=5)
    if result:
        pop_score, recs = result
        print(f"\n🎯 Predicted Popularity Score: {pop_score:.2f}")
        print("\n🔍 Recommended Products:\n")
        print(recs[["name", "rating", "similarity", "popularity"]].to_string(index=False))
