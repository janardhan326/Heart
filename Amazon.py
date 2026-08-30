import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Amazon Reviews Analytics",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------------
# NLTK DOWNLOAD
# ---------------------------------------------------------

@st.cache_resource
def download_nltk_data():

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_data()

# ---------------------------------------------------------
# NLP OBJECTS
# ---------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------
# TEXT PREPROCESSING FUNCTION
# ---------------------------------------------------------

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


# ---------------------------------------------------------
# SAMPLE DATA
# Used when amazon_reviews.csv is not available
# ---------------------------------------------------------

def create_sample_data():

    data = {

        "Product_Name": [
            "Wireless Headphones",
            "Wireless Headphones",
            "Smart Watch",
            "Smart Watch",
            "Laptop",
            "Laptop",
            "Bluetooth Speaker",
            "Bluetooth Speaker",
            "Mobile Phone",
            "Mobile Phone",
            "Wireless Mouse",
            "Wireless Mouse",
            "Keyboard",
            "Keyboard",
            "Power Bank",
            "Power Bank",
            "USB Cable",
            "USB Cable",
            "Earbuds",
            "Earbuds"
        ],

        "Rating": [
            5, 4, 5, 2, 5, 1, 4, 2, 5, 3,
            4, 1, 5, 2, 4, 1, 5, 3, 5, 2
        ],

        "Review_Text": [
            "Excellent headphones with amazing sound quality",
            "Good headphones and comfortable to wear",
            "Amazing smart watch with excellent features",
            "The watch stopped working after a few days",
            "Very good laptop with fast performance",
            "Terrible laptop, very slow and poor quality",
            "The speaker has very good sound quality",
            "Bad speaker and the battery does not last",
            "Excellent phone with great camera",
            "The phone is okay but nothing special",
            "Good mouse and works perfectly",
            "The mouse stopped working very quickly",
            "Excellent keyboard and comfortable typing",
            "Very bad keyboard, keys stopped working",
            "Good power bank and charges quickly",
            "Worst power bank, does not charge properly",
            "Excellent cable and strong quality",
            "The cable is average and sometimes disconnects",
            "Amazing earbuds with clear sound",
            "Bad earbuds and poor sound quality"
        ]
    }

    return pd.DataFrame(data)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

@st.cache_data
def load_data():

    try:

        df = pd.read_csv("amazon_reviews.csv")

        return df, "Amazon CSV Dataset"

    except:

        df = create_sample_data()

        return df, "Built-in Sample Dataset"


df, data_source = load_data()

# ---------------------------------------------------------
# FIND REVIEW COLUMN
# ---------------------------------------------------------

possible_review_columns = [
    "Review_Text",
    "Review",
    "review",
    "review_text",
    "Text",
    "text"
]

review_column = None

for column in possible_review_columns:

    if column in df.columns:

        review_column = column
        break


# ---------------------------------------------------------
# FIND RATING COLUMN
# ---------------------------------------------------------

possible_rating_columns = [
    "Rating",
    "rating",
    "Score",
    "score",
    "Stars",
    "stars"
]

rating_column = None

for column in possible_rating_columns:

    if column in df.columns:

        rating_column = column
        break


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

if review_column is None:

    st.error(
        "Review column not found. "
        "Please use a column such as Review_Text or Review."
    )

    st.stop()


if rating_column is None:

    st.error(
        "Rating column not found. "
        "Please use a column such as Rating."
    )

    st.stop()


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

df = df.dropna(subset=[review_column, rating_column])

df[rating_column] = pd.to_numeric(
    df[rating_column],
    errors="coerce"
)

df = df.dropna(subset=[rating_column])

# Keep ratings between 1 and 5
df = df[
    (df[rating_column] >= 1) &
    (df[rating_column] <= 5)
]

# ---------------------------------------------------------
# CREATE SENTIMENT
# ---------------------------------------------------------

def get_sentiment(rating):

    if rating <= 2:

        return "Negative"

    elif rating == 3:

        return "Neutral"

    else:

        return "Positive"


df["Sentiment"] = df[rating_column].apply(get_sentiment)

# ---------------------------------------------------------
# NLP CLEANING
# ---------------------------------------------------------

with st.spinner("Processing reviews using NLP..."):

    df["Clean_Review"] = df[review_column].apply(clean_text)


# ---------------------------------------------------------
# REMOVE EMPTY REVIEWS
# ---------------------------------------------------------

df = df[df["Clean_Review"].str.strip() != ""]

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_model(text_data, labels):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1
    )

    X = vectorizer.fit_transform(text_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        vectorizer,
        model,
        X_test,
        y_test,
        predictions,
        accuracy
    )


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

try:

    (
        vectorizer,
        model,
        X_test,
        y_test,
        predictions,
        accuracy
    ) = train_model(
        tuple(df["Clean_Review"]),
        tuple(df["Sentiment"])
    )

except ValueError as e:

    st.error(
        "Not enough data to train the model. "
        "Please provide more reviews for each sentiment class."
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🛒 Amazon Reviews")

st.sidebar.write(
    "Amazon Product Reviews Analytics "
    "using NLP and Machine Learning"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔤 NLP Analysis",
        "🤖 Sentiment Prediction",
        "📈 Model Performance",
        "📋 Dataset"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🛒 Amazon Product Reviews Analytics")

    st.subheader(
        "NLP & Machine Learning Based Sentiment Analysis"
    )

    st.write(
        """
        This application analyzes Amazon customer reviews
        using Natural Language Processing and Machine Learning.

        The application performs:

        - Data preprocessing
        - Exploratory Data Analysis
        - NLP text cleaning
        - TF-IDF feature extraction
        - Sentiment classification
        - Machine Learning model evaluation
        - Real-time sentiment prediction
        """
    )

    st.info(
        f"Dataset currently loaded: **{data_source}**"
    )

    st.markdown("### 🔄 Project Workflow")

    st.code(
        """
Amazon Reviews
       ↓
Data Cleaning
       ↓
NLP Preprocessing
       ↓
Stopword Removal
       ↓
Lemmatization
       ↓
TF-IDF
       ↓
Logistic Regression
       ↓
Sentiment Prediction
       ↓
Streamlit Dashboard
        """
    )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title("📊 Amazon Reviews Dashboard")

    # Metrics

    total_reviews = len(df)

    average_rating = df[rating_column].mean()

    positive_reviews = (
        df["Sentiment"] == "Positive"
    ).sum()

    negative_reviews = (
        df["Sentiment"] == "Negative"
    ).sum()

    neutral_reviews = (
        df["Sentiment"] == "Neutral"
    ).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Average Rating",
        f"{average_rating:.2f} ⭐"
    )

    col3.metric(
        "Positive",
        positive_reviews
    )

    col4.metric(
        "Neutral",
        neutral_reviews
    )

    col5.metric(
        "Negative",
        negative_reviews
    )

    st.divider()

    # -----------------------------------------------------
    # RATING DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("⭐ Rating Distribution")

    rating_counts = df[rating_column].value_counts().sort_index()

    fig1, ax1 = plt.subplots()

    ax1.bar(
        rating_counts.index.astype(str),
        rating_counts.values
    )

    ax1.set_xlabel("Rating")

    ax1.set_ylabel("Number of Reviews")

    ax1.set_title("Rating Distribution")

    st.pyplot(fig1)

    # -----------------------------------------------------
    # SENTIMENT DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("😊 Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts()

    fig2, ax2 = plt.subplots()

    ax2.bar(
        sentiment_counts.index,
        sentiment_counts.values
    )

    ax2.set_xlabel("Sentiment")

    ax2.set_ylabel("Number of Reviews")

    ax2.set_title("Sentiment Distribution")

    st.pyplot(fig2)

    # -----------------------------------------------------
    # PRODUCT ANALYSIS
    # -----------------------------------------------------

    if "Product_Name" in df.columns:

        st.subheader("📦 Product-wise Average Rating")

        product_rating = (
            df.groupby("Product_Name")[rating_column]
            .mean()
            .sort_values(ascending=False)
        )

        st.dataframe(
            product_rating.reset_index(),
            use_container_width=True
        )


# =========================================================
# NLP ANALYSIS
# =========================================================

elif page == "🔤 NLP Analysis":

    st.title("🔤 Natural Language Processing Analysis")

    st.write(
        """
        NLP preprocessing converts customer reviews into
        clean text that can be used by the Machine Learning model.
        """
    )

    # -----------------------------------------------------
    # SHOW ORIGINAL AND CLEAN TEXT
    # -----------------------------------------------------

    st.subheader("Original vs Cleaned Reviews")

    display_df = df[
        [review_column, "Clean_Review", "Sentiment"]
    ].head(20)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # REVIEW LENGTH
    # -----------------------------------------------------

    df["Review_Length"] = (
        df[review_column]
        .astype(str)
        .apply(len)
    )

    st.subheader("📝 Review Length Analysis")

    fig3, ax3 = plt.subplots()

    ax3.hist(
        df["Review_Length"],
        bins=20
    )

    ax3.set_xlabel("Review Length")

    ax3.set_ylabel("Number of Reviews")

    ax3.set_title("Distribution of Review Length")

    st.pyplot(fig3)

    # -----------------------------------------------------
    # MOST COMMON WORDS
    # -----------------------------------------------------

    st.subheader("🔤 Most Common Words")

    all_words = " ".join(
        df["Clean_Review"]
    ).split()

    word_frequency = (
        pd.Series(all_words)
        .value_counts()
        .head(20)
    )

    st.dataframe(
        word_frequency.reset_index()
        .rename(
            columns={
                "index": "Word",
                0: "Frequency"
            }
        ),
        use_container_width=True
    )

    fig4, ax4 = plt.subplots()

    ax4.barh(
        word_frequency.index[::-1],
        word_frequency.values[::-1]
    )

    ax4.set_xlabel("Frequency")

    ax4.set_title("Top 20 Most Common Words")

    st.pyplot(fig4)


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

elif page == "🤖 Sentiment Prediction":

    st.title("🤖 Customer Review Sentiment Prediction")

    st.write(
        "Enter a customer review below and the ML model "
        "will predict its sentiment."
    )

    user_review = st.text_area(
        "Enter Customer Review",
        placeholder=(
            "Example: The product is excellent "
            "and I am very happy with my purchase."
        ),
        height=150
    )

    if st.button(
        "🔮 Predict Sentiment",
        use_container_width=True
    ):

        if user_review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            cleaned_review = clean_text(
                user_review
            )

            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            prediction = model.predict(
                review_vector
            )[0]

            probabilities = model.predict_proba(
                review_vector
            )[0]

            confidence = np.max(
                probabilities
            ) * 100

            st.divider()

            if prediction == "Positive":

                st.success(
                    f"😊 Positive Review"
                )

            elif prediction == "Negative":

                st.error(
                    f"😞 Negative Review"
                )

            else:

                st.warning(
                    f"😐 Neutral Review"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "**Cleaned Review:**"
            )

            st.code(
                cleaned_review
            )

            # Probability table

            probability_df = pd.DataFrame(
                {
                    "Sentiment": model.classes_,
                    "Probability": probabilities
                }
            )

            probability_df[
                "Probability"
            ] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.subheader(
                "Prediction Probabilities"
            )

            st.dataframe(
                probability_df,
                use_container_width=True
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.write(
        """
        The model used for sentiment classification is:

        **Logistic Regression**

        Text features are generated using:

        **TF-IDF (Term Frequency-Inverse Document Frequency)**
        """
    )

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    fig5, ax5 = plt.subplots()

    ax5.imshow(cm)

    ax5.set_xlabel("Predicted")

    ax5.set_ylabel("Actual")

    ax5.set_title("Confusion Matrix")

    ax5.set_xticks(
        range(len(model.classes_))
    )

    ax5.set_yticks(
        range(len(model.classes_))
    )

    ax5.set_xticklabels(
        model.classes_
    )

    ax5.set_yticklabels(
        model.classes_
    )

    for i in range(len(cm)):

        for j in range(len(cm[i])):

            ax5.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig5)


# =========================================================
# DATASET
# =========================================================

elif page == "📋 Dataset":

    st.title("📋 Amazon Reviews Dataset")

    st.write(
        f"Dataset Source: **{data_source}**"
    )

    st.write(
        f"Number of rows: **{len(df)}**"
    )

    st.write(
        f"Number of columns: **{len(df.columns)}**"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # Download dataset

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Processed Dataset",
        data=csv,
        file_name="processed_amazon_reviews.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.info(
    """
    Developed using:

    Python
    Pandas
    NumPy
    NLTK
    Scikit-learn
    NLP
    Streamlit
    """
)
