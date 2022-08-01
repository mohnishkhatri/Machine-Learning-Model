#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits.head()


# In[4]:


movies.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[7]:


movies.head()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.dropna(inplace=True)


# In[10]:


movies.isnull().sum()


# In[11]:


movies.duplicated().sum()


# In[12]:


import ast


# In[13]:


def convert(obj):
       list=[]
       for i in ast.literal_eval(obj):
           list.append(i['name'])
       return list


# In[14]:


movies['genres']=movies['genres'].apply(convert)


# 

# In[15]:


movies['genres'].head()


# In[16]:


movies['keywords']=movies['keywords'].apply(convert)


# In[17]:


movies['keywords'].head()


# In[18]:


movies['cast'].head()


# In[19]:


def convert5(obj):
       list=[]
       counter=0
       for i in ast.literal_eval(obj):
           if counter!=5:
               list.append(i['name'])
               counter+=1
           else:
               break
       return list


# In[20]:


movies['cast']=movies['cast'].apply(convert5)


# In[21]:


movies['crew'][0]


# In[22]:


def director(obj):
    list=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            list.append(i['name'])
            break
    return list


# In[23]:


movies['crew']=movies['crew'].apply(director)


# In[24]:


movies['crew'].head()


# In[25]:


movies['overview'].head()


# In[26]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[27]:


movies['overview'].head()


# In[28]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# 

# In[29]:


movies.head()


# In[30]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['crew']+movies['cast']


# In[31]:


movies.head()


# In[32]:


df=movies[['movie_id','title','tags']]


# In[33]:


df['tags']=df['tags'].apply(lambda x:' '.join(x))


# In[34]:


df['tags']=df['tags'].apply(lambda x: x.lower())


# In[35]:


df.head()


# In[36]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[37]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[38]:


df['tags']=df['tags'].apply(stem)


# In[39]:


df['tags'].head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000,stop_words='english')


# In[41]:


vectors=cv.fit_transform(df['tags']).toarray()


# In[42]:


vectors


# In[43]:


cv.get_feature_names()


# In[44]:


from sklearn.metrics.pairwise import cosine_similarity


# In[45]:


cosine_similarity(vectors)


# In[50]:


similarity=cosine_similarity(vectors)


# In[53]:


def recommend(movie):
    movie_index=df[df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(df.iloc[i[0]].title)


# In[54]:


recommend('Avatar')


# In[ ]:




