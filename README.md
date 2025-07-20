# AI-Powered Fashion Recommendation System

## 🧠 Overview

This project is an AI-powered fashion recommendation and popularity prediction system designed for online fashion products. It uses deep learning to understand product images and generate:

* **Similar Product Recommendations** based on image similarity
* **Popularity Prediction** using product ratings and review counts
* **Trend Clustering** of popular vs. lagging designs using unsupervised learning

## 💡 Features

* Upload a fashion product image to receive **top visually similar recommendations**
* Predict how **popular** a product is likely to be
* Run image **clustering** to discover trendy and outdated designs
* Interactive **Streamlit UI** for quick testing and visualization

## 🛠️ Technologies Used

* **TensorFlow / Keras** – deep learning models
* **YOLOv8 (Ultralytics)** – product detection and cropping
* **scikit-learn** – clustering & similarity (KMeans, cosine similarity)
* **Pandas & NumPy** – data processing
* **Streamlit** – frontend for recommendations

## 📸 Sample Output


[Recommendations](https://github.com/anandhu2311c/fashion_recom/blob/777204ae38785a5fce2438e1e257d69566bc3087/images/ss.jpg?raw=true) [Redirecting to Website](https://github.com/anandhu2311c/fashion_recom/blob/777204ae38785a5fce2438e1e257d69566bc3087/images/ss1.jpg?raw=true)

## 🧩 How It Works

### 1. Image Encoding

* Each product image is resized and passed through an encoder CNN to generate a feature vector.
* Encoder models need to be trained on the data

### 2. Similarity-Based Recommendation

* When a new image is uploaded, its vector is compared with others using **cosine similarity**.
* Top-K most similar items are recommended.

### 3. Popularity Prediction

* Popularity is computed from real product

* A regression model is trained to **predict this score** from image encodings.

### 4. Trend Clustering

* KMeans is used to cluster product embeddings.
* Most frequent cluster = **popular trend**
* Least frequent clusters = **lagging styles**

## 🧪 Web Scraping & Dataset Construction

We constructed our dataset by scraping data from major Indian e-commerce platforms:

* **Amazon** – Images of various fashion items (shirts, jeans, kurtas, dresses, etc.), product titles, ratings, and prices
* **Flipkart** – Fashion product images, ratings, review counts, and product URLs
* **Myntra** – Clothing and accessory product images and descriptions

Custom scripts were built to collect and parse product data. Each image was downloaded and passed through a YOLOv8 object detector to crop relevant regions before encoding.

To run web scraping:

* Run the `webscraping.py` script.
* Modify scraping parameters in `scraping_metadata.json` to change sources, keywords, and image limits.

## 🏗️ Model Training Instructions

### 🔁 Train Encoder Model

To train your own image encoder:

* Use the `encoder_training_script.py` file.
* You can run this on **Google Colab (free GPU)** for faster training.

### 📈 Train Popularity Prediction Model

* Uses ratings and review counts to compute a popularity score
* Run the popularity model script after encoding images with the trained encoder

## 📦 Setup Instructions

```bash
# Clone the repo
$ git clone https://github.com/your-username/fashion-ai-recommender.git
$ cd fashion-ai-recommender

# Create and activate virtual environment
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt

# Run Streamlit App
$ streamlit run app.py
```

## 📌 TODO & Improvements

* Integrate **BERT-based sentiment classification** for better popularity estimation
* Sentiment analysis for the popularity
* Extend to **multi-product recommendations**
* Add **real-time web scraping** support
* Add user profile-based personalization

## 🤝 Acknowledgments

* Dataset compiled from Flipkart, Amazon, Myntra, and Pinterest
* YOLOv8 by Ultralytics
* Streamlit for rapid prototyping

---

Built with ❤️ by Anandhu for real-world fashion intelligence 🚀
