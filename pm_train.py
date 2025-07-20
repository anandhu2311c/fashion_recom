import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------- Load Encoder -------------------
encoder_path = "/content/encoder.keras"
encoder = tf.keras.models.load_model(encoder_path, compile=False)

# ------------------- Load Dataset -------------------
df = pd.read_csv("/content/real_fashion_data.csv")

# ------------------- Clean Ratings and Reviews -------------------
df["rating"] = pd.to_numeric(df["rating"], errors='coerce').fillna(3.0)  # neutral rating
if "no_of_reviews" not in df.columns:
    df["no_of_reviews"] = 10  # default review count
df["no_of_reviews"] = pd.to_numeric(df["no_of_reviews"], errors='coerce').fillna(10).astype(int)

# ------------------- Load and Resize Images -------------------
images = []
valid_indices = []

for i in range(len(df)):
    img_path = os.path.join("/content/crop/cropped_tshirts", f"tshirt_{i}_0.jpg")
    if os.path.exists(img_path):
        try:
            img = mpimg.imread(img_path)
            img = resize(img, (320, 192, 3))
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"❌ Error reading {img_path}: {e}")

if len(images) == 0:
    raise ValueError("❌ No valid images found in /content/crop/cropped_tshirts")

print(f"✅ Loaded {len(images)} valid images.")

# ------------------- Match DataFrame with Valid Images -------------------
df = df.loc[valid_indices].reset_index(drop=True)

# ------------------- Encode Images -------------------
image_array = np.array(images)
encodings = encoder.predict(image_array)
df["encodings"] = [e.flatten() for e in encodings]

# ------------------- Compute Popularity -------------------
def pop_met(n, s):
    n = np.array(n)
    s = np.array(s)
    top = s * (15 + n)
    bott = n + 5 * s + 1e-5  # epsilon to avoid divide-by-zero
    return top / bott

df["popularity"] = pop_met(df["no_of_reviews"], df["rating"])

# ------------------- Normalize Popularity -------------------
scaler = MinMaxScaler()
df["popularity"] = scaler.fit_transform(df["popularity"].values.reshape(-1, 1))

# ------------------- Drop NaNs and Prepare Training Data -------------------
df["popularity"].replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=["popularity", "encodings"])
df = df[df["encodings"].apply(lambda x: np.isfinite(x).all())]

X = np.array(df["encodings"].tolist())
y = np.array(df["popularity"]).reshape(-1, 1)

# ------------------- Build Model -------------------
model = Sequential([
    Dense(1920, input_dim=1920, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

# ------------------- Train Model -------------------
model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

# ------------------- Save Model and Results -------------------
model.save("pm_model_components/pm_model.keras")
df_sorted = df.sort_values(by='popularity', ascending=False)
df_sorted.to_csv("df_sorted.csv", index=False)

print("Model training complete.")
print("Sorted DataFrame saved as df_sorted.csv.")
print("PM model saved to pm_model_components/pm_model.keras.")
