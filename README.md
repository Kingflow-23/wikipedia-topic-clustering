# Wikipedia Topic Clustering and Visualization

This project scrapes text data from Wikipedia pages, processes the data using TF-IDF vectorization, and visualizes the clustering of topics using UMAP for dimensionality reduction and KMeans clustering.

## Project Overview

The goal of this project is to fetch content from various Wikipedia pages on different topics, clean and preprocess the text data, vectorize the content using the Term Frequency-Inverse Document Frequency (TF-IDF) method, and visualize the relationships between these topics using dimensionality reduction techniques. The topics are clustered using KMeans and visualized with UMAP projections.

### Primary Steps in the Workflow:

---

### 1. Scraping Wikipedia Pages

**Objective:**  
The first step is to gather content from Wikipedia pages on various topics of interest. 

- **How It Works:**
  - The script uses the `requests` library to send HTTP requests to the Wikipedia API (or directly scrape HTML content from the pages).
  - The `beautifulsoup4` library is used to parse the HTML structure and extract only the relevant text data (the content section of each article).
  - In the notebooks I try an approach of scraping using wikipedia-api and also a direct scraping approach using beautifulsoup4.

**Topics Example:**  
- Artificial Intelligence
- Computer Science
- Machine Learning
- Data Science

Once the data is scraped, it is stored in a structured format (e.g., a Pandas DataFrame) with columns for the topic title and the corresponding text content.

---

### 2. Text Preprocessing

**Objective:**  
Once the content is scraped, the next step is to preprocess and clean the text data to make it suitable for analysis.

- **How It Works:**
  - The text is cleaned by removing unwanted characters like punctuation and numbers, stripping unnecessary whitespace, and converting the text to lowercase for uniformity.
  - Regular expressions (`re` module) are used to filter out non-alphabetic characters, leaving only the relevant words.
  - Additional text normalization techniques may include tokenization (splitting text into individual words) and removing stopwords (common words like “the,” “and,” etc., that don’t add meaningful context).
  
**Result:**  
At the end of preprocessing, you’ll have a clean list of text data that is ready for vectorization.

---

### 3. TF-IDF Vectorization

**Objective:**  
The next step is to convert the text data into numerical format, which will enable the machine learning algorithm (KMeans) to work with it.

- **How It Works:**
  - The TF-IDF (Term Frequency-Inverse Document Frequency) method is applied to the preprocessed text.
  - TF-IDF assigns a weight to each word based on how frequently it appears in a document relative to its frequency across all documents. This helps highlight the most important terms for each topic.
  - The `TfidfVectorizer` from `scikit-learn` is used to convert the preprocessed text into a sparse matrix of numerical values.
  
**Result:**  
A TF-IDF matrix where each row corresponds to a document (Wikipedia page), and each column corresponds to a word (term) with its corresponding weight (importance).

---

### 4. Clustering Using KMeans

**Objective:**  
With the TF-IDF matrix in hand, we now want to group similar Wikipedia topics into clusters based on their content.

- **How It Works:**
  - KMeans is an unsupervised machine learning algorithm used for clustering.
  - The algorithm divides the dataset into a predefined number of clusters (K). It does this by iteratively assigning data points (in this case, Wikipedia topics) to clusters based on the closest centroids.
  - The KMeans algorithm in `scikit-learn` is used, with the number of clusters (K) defined based on prior knowledge or experimentation.
  
**Result:**  
Each Wikipedia topic is assigned to one of the K clusters, grouping similar topics together.

---

### 5. Dimensionality Reduction Using UMAP

**Objective:**  
After clustering the topics, we need to visualize the results. Since the TF-IDF matrix is high-dimensional, we need to reduce its dimensionality to plot the data on a 2D graph.

- **How It Works:**
  - UMAP (Uniform Manifold Approximation and Projection) is used for dimensionality reduction, which preserves both the local and global structures of the data.
  - UMAP projects the high-dimensional TF-IDF matrix into a 2D space, which makes it easier to visualize and interpret the clusters.
  
**Result:**  
A 2D projection of the clustered topics, where each point on the plot represents a Wikipedia topic, and the color or shape of the points indicates which cluster they belong to.
A 3d version and interactive one (with plotly) can be found in the api_project_3d.ipynb notebook

---

### 6. Saving and Displaying the Plot

**Objective:**  
Finally, we want to visualize the clustering results in a meaningful way and save the visualization for future reference.

- **How It Works:**
  - The `matplotlib` library is used to create a scatter plot.
  - Each point in the plot represents a Wikipedia topic, and the points are color-coded according to the cluster they belong to.
  - A timestamp is added to the filename of the plot to make each generated plot unique.
  
**Result:**  
- A PNG file of the 2D visualization is saved to the disk with a filename in the format `scraping_wikipedia_result_YYYY-MM-DD_HH-MM-SS.png`.
- The plot is displayed on the screen so you can visually inspect how the topics are clustered.

---

### Output:

- **Saved Plot:** A PNG file of the 2D scatter plot showing the clustering of Wikipedia topics.
- **Visualization:** A 2D plot where similar topics are grouped together, making it easy to see which topics are related based on their textual content.

Example of a saved file name:  
`scraping_wikipedia_result_YYYY-MM-DD_HH-MM-SS.png`

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Kingflow-23/wikipedia-topic-clustering.git
```

```bash
pip install requests beautifulsoup4 pandas seaborn matplotlib scikit-learn umap-learn wikipedia-api plotly
```

or 

```bash
pip install -r requirements.txt
```