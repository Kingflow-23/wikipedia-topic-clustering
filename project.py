import re
import umap
import time
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config import *

not_working_pages = []


def get_wikipedia_page_text(title: str):
    """
    Fetches and returns the cleaned text content of a Wikipedia page for a given title.
    The function scrapes the HTML content using BeautifulSoup, extracts text from paragraph elements,
    and cleans the text using the `cleaning_text()` function.

    Arguments:
        title (str): The title of the Wikipedia page.

    Returns:
        str: Cleaned text content of the page, or an empty string if an error occurs.
    """
    global not_working_pages

    base_url = f"https://en.wikipedia.org/wiki/{title}"
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract text from all paragraphs
        paragraphs = soup.find_all("p")
        page_text = " ".join([para.get_text() for para in paragraphs])
        return cleaning_text(page_text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {title}: {e}")
        not_working_pages.append(title)
        return ""


def cleaning_text(text: str):
    """
    Cleans extracted Wikipedia page text by removing punctuation, non-alpha characters,
    and multiple spaces while converting to lowercase.

    Arguments:
        text (str): Raw text extracted from the Wikipedia page.

    Returns:
        str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\n", "", text)  # Remove newline characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"[.,;:\'\"“”’]", "", text)  # Remove punctuations
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alpha characters
    return text


def vectorize(data: pd.DataFrame):
    """
    Vectorizes the corpus using the TF-IDF method, creating a feature matrix where each row corresponds
    to a document, and each column corresponds to a word in the vocabulary.

    Arguments:
        data (pd.DataFrame): DataFrame of documents to vectorize.

    Returns:
        pd.DataFrame: TF-IDF weighted term-document matrix as a DataFrame.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer object for future use.
    """
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")

    # Fit and transform the documents
    X = vectorizer.fit_transform(data["Text"])

    # Convert the sparse matrix into a dense DataFrame
    tfidf_df = pd.DataFrame(
        X.toarray(), columns=vectorizer.get_feature_names_out(), index=data.index
    )

    return tfidf_df, vectorizer


def vectorize_new_topic(new_topic_texts: list, vectorizer: object):
    """
    Vectorizes new topics using the already fitted TF-IDF vectorizer from the original corpus.

    Arguments:
        new_topic_texts (list): List of texts for new topics to be vectorized.
        vectorizer (TfidfVectorizer): The pre-fitted TF-IDF vectorizer.

    Returns:
        new_topic_vector (sparse matrix): TF-IDF matrix for the new topics.
        features (array): List of feature names (words) corresponding to the TF-IDF vectors.
    """
    new_topic_vectors = vectorizer.transform(new_topic_texts)

    return new_topic_vectors, vectorizer.get_feature_names_out()


def project_and_visualize_clusters(
    original_data: pd.DataFrame,
    new_topic_vectors,
    index: list,
    new_topic_labels: list,
    features: list,
    n_clusters: int =8,
):
    """
    Combines the original data and new topics, applies UMAP for dimensionality reduction,
    clusters the combined data using k-means, and visualizes the results using a scatter plot.

    Arguments:
        original_data (pd.DataFrame): The TF-IDF matrix of the original corpus.
        new_topic_vectors (sparse matrix): The TF-IDF vectorized data for new topics.
        index (list): List of labels or categories for the original data.
        new_topic_labels (list): Labels for the new topics.
        features (list): Feature names (words) corresponding to the combined data.
        n_clusters (int): Number of clusters for k-means clustering.
    """
    # Combine original data with the new topic
    new_topic_df = pd.DataFrame(
        new_topic_vectors.toarray(), columns=features, index=new_topic_labels
    )
    combined_data = pd.concat([original_data, new_topic_df])

    # Apply UMAP for dimensionality reduction on combined data
    umap_model = umap.UMAP(n_components=2, random_state=42)
    tfidf_2d_combined = umap_model.fit_transform(combined_data)

    # Run k-Means clustering on the combined UMAP-reduced data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_2d_combined)

    # Create a DataFrame for visualization
    umap_df_combined = pd.DataFrame(tfidf_2d_combined, columns=["UMAP1", "UMAP2"])
    umap_df_combined["category"] = index + new_topic_labels
    umap_df_combined["cluster"] = clusters

    # Plot UMAP with the new topic and updated clusters
    plt.figure(figsize=(10, 8))
    scatter_plot = sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="cluster",
        palette=sns.color_palette("hsv", n_clusters),
        data=umap_df_combined,
        legend="full",
    )

    # Add annotations (True Class) to the points
    for i in range(umap_df_combined.shape[0]):
        plt.text(
            umap_df_combined["UMAP1"].iloc[i],  # x-coordinate
            umap_df_combined["UMAP2"].iloc[i],  # y-coordinate
            umap_df_combined["category"].iloc[i],  # True class or label
            fontsize=9,
            color="black",
            ha="right",
            va="bottom",
        )

    handles, labels = scatter_plot.get_legend_handles_labels()
    custom_labels = [f"Cluster {i}" for i in range(n_clusters)]
    plt.legend(handles=handles, labels=custom_labels, title="Clusters")

    plt.title("2D UMAP Projection with New Topic and Updated Clusters")

    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{OUTPUT_DIR}/scraping_wikipedia_result_2d_{formatted_time}.png"

    plt.savefig(filename, format="png")


def main():

    categories = {
        "AI": AI_PAGES,
        "CS": COMPUTER_SCIENCE_PAGES,
        "Law": LAW_PAGES,
        "Literature": LITERATURE_PAGES,
        "Philosophy": PHILOSOPHY_PAGES,
        "History": HISTORY_PAGES,
        "Sports": SPORTS_PAGES,
        "Music": MUSIC_PAGES,
    }

    print("1 - Start data scraping...")
    
    documents = {}

    for category, pages in categories.items():
        documents[category] = [
            (title, get_wikipedia_page_text(title))
            for title in pages
            if get_wikipedia_page_text(title) != ""
        ]

    data = []

    for category, pages in documents.items():
        for title, text in pages:
            data.append({"Category": category, "Title": title, "Text": text})

    data = pd.DataFrame(data).set_index("Category")

    print("Data scraping finished.\n")

    # Vectorize the documents using TF-IDF
    print("2 - Start vectorizing data...")
    tf_idf, vectorizer = vectorize(data)
    print("Vectorizing finished.\n")

    # Get texts and index for new topics.
    print("3 - Start getting new topics...")
    test_topic_texts = [
        get_wikipedia_page_text(new_topic_title)
        for new_topic_title in TEST_TOPIC_PAGES
        if get_wikipedia_page_text(new_topic_title) != ""
    ]
    test_topic_labels = [f"{title}" for title in TEST_TOPIC_PAGES]
    print("Getting new topics finished.\n")

    # Vectorize the new corpus
    print("4 - Start vectorizing new data...")
    new_topic_vectors, features = vectorize_new_topic(test_topic_texts, vectorizer)
    print("Vectorizing new data finished.\n")

    # Project and visualize clusters
    print("5 - Start projecting and visualizing clusters...")
    project_and_visualize_clusters(
        tf_idf,
        new_topic_vectors,
        index=data.index.tolist(),
        new_topic_labels=test_topic_labels,
        features=features,
    )
    print("Projecting and visualizing clusters finished.\n")
    print("Done. ✅")
    
if __name__ == "__main__":
    main()
