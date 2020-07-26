# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:20:13 2020

@author: KayDee
"""
from sklearn.feature_extraction.text import CountVectorizer
set = ["Jaipur Ajmer Jaipur","Ajmer Ajmer Jaipur"]

vector = CountVectorizer()
vector_fit = vector.fit_transform(set)
array = vector_fit.toarray()
print(array)
print(vector_fit)

from sklearn.metrics.pairwise import cosine_similarity

cos = cosine_similarity(array)
print(cos)

