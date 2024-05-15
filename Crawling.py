#!/usr/bin/env python
# coding: utf-8

# # Final Project - Earlier crawling version
# ### EN.601.666 Information Retrieval and Web Agents
# ### Isabel Dinan idinan1@jh.edu

# In[80]:


# !pip install pdfplumber
# !pip install nltk gensim matplotlib
# !pip install pyLDAvis


# In[81]:


import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
import pdfplumber
from io import BytesIO

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from gensim import corpora, models
import pandas as pd

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


# In[82]:


download_dir = os.path.join(os.getcwd(), "pdfs")
FETCHED = []

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[95]:


def fetch_paper_links(category, max_papers=50):
    """
    Fetch links to paper abstracts from the arXiv category listing page.
    """
    base_url = 'https://export.arxiv.org/list'
    url = f"{base_url}/{category}/recent"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = [f"https://export.arxiv.org{a['href']}" for a in soup.find_all('a', title='Abstract')][:max_papers]
    
    time.sleep(15)
    
    return links

def fetch_paper_details(url):
    """
    Fetch details of a paper given its abstract page URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1', class_='title').text.replace('Title: ', '').strip()
    authors = ', '.join([author.text.strip() for author in soup.find_all('div', class_='authors')])
    abstract = soup.find('blockquote', class_='abstract').text.replace('Abstract: ', '').strip()
    meta_info = soup.find_all('div', class_='dateline')
    submission_date = meta_info[0].text.strip() if meta_info else 'Not provided'
    # submission_date = soup.find(text='Submitted').next.strip()
    
    pdf_link = url.replace('/abs/', '/pdf/') + '.pdf'
    pdf_response = requests.get(pdf_link)
    # if not os.path.exists(download_dir):
    #     os.makedirs(download_dir)
    # pdf_filename = os.path.join(download_dir, pdf_link.split('/')[-1])
    # with open(pdf_filename, 'wb') as f:
    #     f.write(pdf_response.content)
    # print(f"Downloaded PDF to {pdf_filename}")
    pdf_file = BytesIO(pdf_response.content)
    try:
        with pdfplumber.open(pdf_file) as pdf:
            pdf_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        print("saved pdf for: ", pdf_link)
    except Exception as e:
        print(f"PDFSyntaxError encountered for URL: {pdf_link}. Skipping this file.")
        pdf_text = ""
    
    time.sleep(15)
    
    return {'title': title, 'authors': authors, 'abstract': abstract, 'submission_date': submission_date, 
            'pdf_text': pdf_text, 
            'url': url}

def crawl_arxiv(category, max_papers=50):
    """
    Main function to crawl arXiv for papers in a specific category.
    """
    links = fetch_paper_links(category, max_papers)
    papers = []
    for link in links:
        if link not in FETCHED:
            FETCHED.append(link)
            paper = fetch_paper_details(link)
            papers.append(paper)
            print(f"Fetched: {paper['title'][:30]}...")
    return papers


# In[93]:


category = 'stat.ML'


# In[94]:


ml_papers = crawl_arxiv(category, max_papers=50)


# In[ ]:


papers_df = pd.DataFrame(ml_papers)
print(papers_df.head())  # Show the first few entries


# In[ ]:


papers_df.to_csv('machine_learning_papers_1.csv', index=False)


# In[72]:


def get_custom_stopwords():
    stop_words = set(stopwords.words('english'))
    custom_stops = ["http", "https", "et", "al", "url", "www", "com", "edu", 
                    "fig", "figure", "table", "pdf", "citation", "reference", 
                    "estimator", "method", "site", "meta", "using", "able", "never",
                    "often", "mentioned", "others", "accordingly", "otherwise",
                    "accross", "must", "much", "moreover", "might", "doi", "get"]
    stop_words.update(custom_stops)
    return stop_words


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if len(word) > 1]
    stop_words = get_custom_stopwords()
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def perform_topic_modeling(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, alpha='auto', eta='auto')
    topics = lda_model.print_topics(num_words=5)
    return topics


# In[73]:


papers_df['processed_text'] = papers_df['pdf_text'].apply(preprocess_text)
topics = perform_topic_modeling(papers_df['processed_text'].tolist())
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {topic}")


# In[ ]:


all_words = [word for text in papers_df['processed_text'] for word in text]
word_freq = Counter(all_words)
print(word_freq.most_common(20))


# In[ ]:




