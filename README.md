# Topic-clustering-analysis
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import gensim.downloader as api
import gensim.downloader
import umap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# vectors = gensim.downloader.load('word2vec-google-news-300')
vectors = api.load("word2vec-google-news-300")
def text_to_vec(text):
    words = text.lower().split()
    vecs = [vectors[word] for word in words if word in vectors]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)
# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
def preprocess_text(texts):
    # Initialization tool
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for text in texts:
        # Lowercase+Punctuation Removal
        text = ''.join([c.lower() for c in text if c.isalpha() or c == ' '])
        # Word segmentation and form reduction
        tokens = [lemmatizer.lemmatize(word) for word in text.split()
                  if word not in stop_words and len(word) > 3]
        processed_texts.append(' '.join(tokens))
    return processed_texts
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
data_abstract=pd.read_excel("data/savedrecs.xls")
data_abstract.dropna(inplace=True)
#Vectorization
n=len(data_abstract)
X_vec=np.zeros([n,300])
for i in range(n):
    X_vec[i]=text_to_vec(data_abstract.values[i,0])
K=10
kmeans=KMeans(n_clusters=K,random_state=0)
kmeans.fit(X_vec)
labels=kmeans.labels_
# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2,random_state=42)
X_umap = reducer.fit_transform(X_vec)
for k in range(K):
    plt.scatter(X_umap[labels==k,0],
                X_umap[labels==k,1],
                s=7,
                label="cluster {}".format(k+1)
                )
plt.legend(loc=(1.03,0.2))
plt.savefig("figures/umap scatter plot-k={}.png".format(K),dpi=600,bbox_inches='tight')
plt.show()
num_topic_words = 10
top_words_list = []
for k in range(K):
    texts = data_abstract.iloc[labels == k].values.reshape(-1, ).tolist()
    processed_texts = preprocess_text(texts)
    vectorizer = CountVectorizer(max_features=100)
    X = vectorizer.fit_transform(processed_texts)
    # Set to extract 1 theme
    lda = LatentDirichletAllocation(
        n_components=1,  
        random_state=42,
        learning_method='online',
        max_iter=100
    )
    lda.fit(X)
    # Get vocabulary list
    feature_names = vectorizer.get_feature_names_out()
    # Obtain the distribution of keywords
    topic_words = lda.components_[0]
    top_word_indices = topic_words.argsort()[-num_topic_words:][::-1]
    top_words = [feature_names[i] for i in top_word_indices]
    top_words_list.append(" ".join(top_words))
    print(top_words)
top_words_df = pd.DataFrame({"cluster": ["cluster {}".format(k + 1) for k in range(K)],
                             "topic words": top_words_list})
