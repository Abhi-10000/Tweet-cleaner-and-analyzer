# Hybrid-Intelligence Social Media Analysis System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-purple.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a full-stack, hybrid-intelligence data science project that transforms a massive, raw tweet dataset (Sentiment140) into a real-time, interactive analysis application.

It intelligently combines a fast, traditional machine learning pipeline for large-scale data enrichment with an advanced LLM agent (Groq) for on-demand, qualitative reasoning. The entire system is deployed as an interactive web application using Streamlit.

---


## 📸 Application Screenshot

<img width="846" height="751" alt="image" src="https://github.com/user-attachments/assets/2b05a5be-88a3-4ac4-b121-8af8301d71c2" />
<img width="1024" height="1014" alt="image" src="https://github.com/user-attachments/assets/887ccb5e-f30c-43e1-b7c7-d423025d8935" />



---

## 🚀 Core Features

* 🧠 **Unsupervised Topic Modeling:** The system automatically discovers 20 distinct, relevant themes from the tweet data. It uses `SentenceTransformer` embeddings and `KMeans` clustering, replacing a rigid, pre-defined model with one that learns from the data itself.
* ⚡ **Hybrid-Intelligence Pipeline:** This project uses a two-phase architecture.
    1.  **Offline:** A fast `scikit-learn` pipeline pre-processes and analyzes 20,000+ tweets for sentiment and theme.
    2.  **Online:** An advanced Groq LLM (via `LangChain`) analyzes a small, *pre-synthesized report* to provide deep, qualitative insights in seconds.
* 📈 **Advanced Engagement Metrics:** This app moves beyond simple tweet counts. It introduces a custom **"Impact Score"** that weighs a topic's total engagement (likes + retweets) by its sentiment, providing a much deeper insight into a topic's true influence.
* 📊 **Interactive Web Application:** The entire system is wrapped in a clean, user-friendly web app built with `Streamlit`, allowing any user to perform complex analysis by simply typing in a query.

---

## 🛠️ Technology Stack

* **Frontend/UI:** Streamlit
* **Backend & Data Processing:** Python, Pandas, NumPy
* **Machine Learning (Traditional):** scikit-learn (Logistic Regression, KMeans, TfidfVectorizer)
* **NLP & Semantic Search:** SentenceTransformers (all-MiniLM-L6-v2), NLTK
* **LLM Integration:** LangChain, Groq
* **Database:** SQLite

---

## 🏗️ System Architecture & Workflow

The system is split into two distinct phases: **Offline Enrichment** (run once to build the database) and **Online Analysis** (run in real-time by the user).

![Uploading image.png…]()

*made using nano banana


### Phase 1: Offline Enrichment (The "Factory")

This is the data engineering pipeline that builds our core asset, the `tweet_analysis.db`.

1.  **Data Loading:** Raw `sentiment140.csv` is loaded.
2.  **Model Training:**
    * `train_sentiment_model.py`: Trains a `LogisticRegression` model on TF-IDF vectors to classify sentiment (0=Neg, 1=Pos). Saves `sentiment_model.pkl`.
    * `train_cluster_model.py`: Trains a `KMeans` model on `SentenceTransformer` embeddings to discover 20 themes. Saves `kmeans_model.pkl` and `cluster_centroids.npy`.
3.  **Database Population (`enrich_db.py`):**
    * A loop runs 20,000+ times. For each raw tweet, it:
    * **Cleans** the text using `text_preprocessor_tool.py`.
    * **Predicts** sentiment using the loaded sentiment model.
    * **Predicts** the theme (Cluster ID) using the loaded cluster model.
    * **Simulates** `likes` and `retweets` data using NumPy.
    * **Saves** all this data (raw text, cleaned text, sentiment, theme ID, likes, retweets) to the `tweet_analysis.db`.

### Phase 2: Online Analysis (The "App")

This is the real-time user-facing application, `app.py`.

1.  **Startup:** The Streamlit app loads. The `SentenceTransformer` model and `cluster_centroids.npy` are loaded into cache (`@st.cache_resource`).
2.  **User Input:** A user enters a query (e.g., "my boss is annoying").
3.  **Semantic Search:**
    * The user's query is converted into a vector embedding.
    * This vector is compared against the 20 cluster centroids to find the closest match (e.g., Cluster 18: "Discussing Work & Jobs").
4.  **Data Retrieval:**
    * The app queries the `tweet_analysis.db` for all tweets in the matched cluster (`SELECT * WHERE theme = 18`).
    * The results are sorted by `likes DESC` to find the most influential tweets.
5.  **Quantitative Analysis:**
    * The app instantly calculates all metrics from this DataFrame:
        * Total Tweets
        * Sentiment Distribution (by count)
        * Positive/Negative Engagement (by likes/retweets)
        * The final **"Impact Score"**
6.  **Qualitative Synthesis (`agent.py`):**
    * A detailed, synthesized prompt (containing the user query, matched theme, all quantitative stats, and the text of the *Top 10 Most-Liked Tweets*) is sent to the Groq LLM.
7.  **Final Report:** The Streamlit UI displays the full quantitative report and the qualitative JSON summary from the LLM.

---

## 🔧 Setup & Installation

Follow these steps to run the project locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    * Create a file named `.env` in the root directory.
    * Add your Groq API key to it:
        ```
        GROQ_API_KEY="your-api-key-here"
        ```

5.  **Download NLTK data:**
    ```bash
    python download_nltk.py
    ```

6.  **Run the Offline Training & Enrichment Pipeline:**
    * **Important:** You must run these scripts in order. The last step will take several minutes.

    ```bash
    # 1. Train the sentiment model
    python train_sentiment_model.py

    # 2. Train the unsupervised theme model
    python train_cluster_model.py

    # 3. Build the database (This will take time)
    python enrich_db.py
    ```
    * *(Optional)*: Run `inspect_clusters.py` to see the themes you've created.

7.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open to the application.

---

## 🔮 Future Improvements

* **Live Data Pipeline:** Replace the static `enrich_db.py` script with a real-time data ingestion pipeline (e.g., using Kafka or Twitter's API) to analyze new tweets as they happen.
* **Real Engagement Data:** Use an API to fetch *actual* like/retweet counts instead of simulating them, making the "Impact Score" 100% accurate.
* **Advanced Sentiment Model:** Replace the `LogisticRegression` model with a fine-tuned transformer model (like BERT) for more nuanced and accurate sentiment analysis.
* **Robust Database:** Upgrade from `SQLite` to a production-grade database like `PostgreSQL` for better scalability.



