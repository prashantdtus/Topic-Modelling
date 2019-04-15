#!/usr/bin/env python
# coding: utf-8

# # Topic Modelling in python - data source is a textfile

# In[1]:


# coding: utf-8 
#encoding=utf-8
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
import unicodedata
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


# ### Set working directory ( only if not using a vitual environment)

# In[2]:



import os
os.chdir('/home/visa/Topic')


# In[3]:


df = pd.DataFrame()
f = open("polarity_pos.txt", "r+", encoding="latin1")
txt = f.readlines()


# In[4]:


imd = pd.DataFrame(txt, columns=["comments"])
imd["row"] = imd.index


# In[5]:


imd.head()


# In[6]:


stop_words = nltk.corpus.stopwords.words('english')
extended_stopwords = ['\'ll','\'d','\'m','\'re','\'s','\'ve','ca n\'t','r','n\'t','ca','see','get','movies','movie','go','say','come','many','another','could','would','made','really','want','even','odd','films','plot','ever','actually','also','movie','film']
stops = stop_words + extended_stopwords


# #### Check with stemming and lemmatization to clean the data

# In[8]:


text = " this is a test for stemmer and stemming "

from nltk.tokenize import word_tokenize
tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
stm = PorterStemmer()
lemm = WordNetLemmatizer()

#tokens = [stm.stem(w) for w in tokens]
tokens = [lemm.lemmatize(w) for w in tokens]
#tokens = lemm.lemmatize(tokens)
print(tokens)


# #### Define function to tokenize and lemmatize the data

# In[7]:


def tokenize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stm = PorterStemmer()
    lemm = WordNetLemmatizer()
    #tokens = [stm.stem(w) for w in tokens]
    tokens = [lemm.lemmatize(w) for w in tokens]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    #import pdb;pdb.set_trace()
    return filtered_tokens


# #### Setiment Analysis with textblob 

# In[13]:


imd['polarity'] = imd.comments.apply(lambda s: TextBlob(s).sentiment.polarity)


# In[15]:


imd.head()
imd.tail()


# #### Document term matrix with TF-IDF values 

# In[16]:


term_idf_vectorizer       = TfidfVectorizer(max_df=0.99, max_features=2000,min_df=0.005, stop_words=stops, use_idf=True, tokenizer=tokenize, ngram_range=(1,1))
get_ipython().run_line_magic('time', 'term_idf_matrix     = term_idf_vectorizer.fit_transform(imd.comments)')
term_idf_feature_names    = term_idf_vectorizer.get_feature_names()
term_idf_matrix.shape


# In[17]:


term_idf_feature_names


# In[18]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# #### Topic Modelling using LDA 

# In[19]:


lda = LatentDirichletAllocation(n_topics=5, max_iter=10,learning_method='online',learning_offset=10.,random_state=1)
get_ipython().run_line_magic('time', 'lda.fit(term_idf_matrix)')
print("\nTopics using Latent Dirichlet Allocation model with Term frequencies: \n")
print_top_words(lda, term_idf_feature_names, 10)


# #### Avoid noise in unigrams, TFIDF matrix on bigrams and trigrams

# In[22]:


term_idf_vectorizer       = TfidfVectorizer(max_df=0.99, max_features=2000,min_df=0.0005, stop_words=stops, use_idf=True, tokenizer=tokenize, ngram_range=(2,3))
get_ipython().run_line_magic('time', 'term_idf_matrix     = term_idf_vectorizer.fit_transform(imd.comments)')
term_idf_feature_names    = term_idf_vectorizer.get_feature_names()
term_idf_matrix.shape


# #### Topic Modelling using LDA ( with bigrams and trigams)

# In[23]:


lda = LatentDirichletAllocation(n_topics=5, max_iter=10,learning_method='online',learning_offset=10.,random_state=1)
get_ipython().run_line_magic('time', 'lda.fit(term_idf_matrix)')
print("\nTopics using Latent Dirichlet Allocation model with Term frequencies: \n")
print_top_words(lda, term_idf_feature_names, 10)


# #### Topic modeling using NMF 

# In[25]:


# Fit the NMF model
get_ipython().run_line_magic('time', 'nmf = NMF(n_components=5, random_state=1,alpha=.1, l1_ratio=.5).fit(term_idf_matrix)')
print("\nFitting the Non-negative Matrix Factorization model with tf-idf features: \n")
print_top_words(nmf, term_idf_feature_names, 10)

