# Hybrid-Intelligence Social Media Analysis

This is a full-stack NLP and data science project that analyzes a large tweet dataset. It combines a fast, traditional ML pipeline with a modern LLM agent to provide deep, real-time insights into 20 different unsupervised-learning-driven themes.

## 🚀 Features
* **Unsupervised Topic Modeling:** Uses K-Means on `SentenceTransformer` embeddings to discover 20 distinct themes.
* **Hybrid AI Pipeline:** Pre-classifies sentiment with `scikit-learn` and uses a Groq LLM for final qualitative summaries.
* **"Impact Score" Metric:** A custom data-driven metric that measures a topic's influence by weighing sentiment and engagement.
* **Interactive UI:** A full web application built with Streamlit.

## 🛠️ How to Run
1.  Clone this repository.
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  (You will need to run the training scripts first - *add your notes here*)
4.  Launch the app:
    ```bash
    streamlit run app.py
    ```