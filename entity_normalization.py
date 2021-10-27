# -*- coding: utf-8 -*-
"""
Entity Normalization for following entities seperately:
  Company Name
  Company Address
  Serial Numbers
  Physical Goods
  Locations
"""
"""
Dependencies:
!pip install fuzzywuzzy
!pip install python-Levenshtein
!pip install recordlinkage
!pip install nltk
!pip install gensim
Tested on colab
"""

import nltk
import gensim
import string
import itertools
import numpy as np
import pandas as pd
import recordlinkage
import gensim.downloader as api
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz, process
from recordlinkage.preprocessing import clean


nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Word2vec pretrained. Takes considerable time
wv = api.load('word2vec-google-news-300')


# Basic preprocessing functions
def tokenize(text):
  tokens = nltk.word_tokenize(text)
  return tokens

def removeStopwords(text):
  tokens = tokenize(text)
  result = ' '.join([token for token in tokens if token not in stop_words])
  return result

def removePunctuation(text):
  tokens = tokenize(text)
  result = ' '.join([token for token in tokens if token not in string.punctuation])
  return result

def removeSpaces(text):
  # spaces = [' ', '\t', '\n']
  # result = ''.join([ch for ch in text if ch not in spaces])
  result = ''.join(text.split())
  return result

"""
Using fuzzy string matching. Try to find most similar string
"""
def companyNameNormalisation(data, thresh=0.5):  
  unique_items = set()
  data['company_name'] = data['company_name'].str.lower()
  data['company_name'] = data['company_name'].apply(removePunctuation)
  print(data)
  duplicates = {}
  choices = []
  for cname in data['company_name']:
    if len(choices) >= 1:
      matchedName , score = process.extractOne(cname, choices)
      score /= 100.0
      print('"{}" matched with "{}" at confidence of:\t {}'.format(cname, matchedName, score))
      if score >= thresh:
        continue
    choices.append(cname)
    unique_items.add(cname)
  print('Unique company names list:\n {}'.format(unique_items))
  return unique_items


"""
Using levenshtein distance with high threshold
"""
def serialNumberNormalisation(data):
  indexer = recordlinkage.Index()
  indexer.full()
  candidate_pairs = indexer.index(data)
  comp = recordlinkage.Compare()
  data['serial_number'] = data['serial_number'].str.lower()
  data['serial_number'] = data['serial_number'].apply(removePunctuation)
  data['serial_number'] = data['serial_number'].apply(removeSpaces)
  print(data.head(10))
  comp.string('serial_number', 'serial_number', method='levenshtein', threshold=0.9, label='serial_number')
  features  = comp.compute(candidate_pairs, data, data)
  print(features)
  common_entries = features[features.sum(axis=1) > 0]
  print(common_entries)


def getMultipleWordsSimilarity(sent1, sent2):
  # Uses word2vec pretrained  
  tokens1 = tokenize(sent1)
  tokens2 = tokenize(sent2)
  similarity_arr = []
  for w1 in tokens1:
    for w2 in tokens2:
      try:
        similarity_arr.append(wv.similarity(w1, w2))
      except:
        # For simplicity ignoring case which is not found in word2vec
        pass
  # Using mean over all word pairs simialrity
  return np.mean(similarity_arr)

"""
Using word2vec and cosine similarity for similarity
"""
def physicalGoodsNormalisation(data, threshold=0.3):
  unique_items = set()
  data['physical_good'] = data['physical_good'].str.lower()
  data['physical_good'] = data['physical_good'].apply(removeStopwords)
  subsets = itertools.combinations(data['physical_good'], 2)
  duplicates = {}
  for w1, w2 in subsets:
    w1 = w1.lower()
    w2 = w2.lower()
    similarity = getMultipleWordsSimilarity(w1, w2)
    print('%r\t%r\t%.2f' % (w1, w2, similarity))
    if similarity >= threshold:
      duplicates[w2] = 1
    if w1 in duplicates:
      continue
    unique_items.add(w1)
  print('Unique physical goods list:\n {}'.format(unique_items))
  return unique_items

"""
Using word2vec and cosine similarity for similarity
"""
def locationNormalisation(data, threshold=0.5):
  unique_items = set()
  duplicates = {}
  data['location'] = data['location'].str.lower()
  data['location'] = data['location'].apply(removeStopwords)
  subsets = itertools.combinations(data['location'], 2)
  for w1, w2 in subsets:
    w1 = w1.lower()
    w2 = w2.lower()
    similarity = getMultipleWordsSimilarity(w1, w2)
    print('%r\t%r\t%.2f' % (w1, w2, similarity))
    if similarity >= threshold:
      duplicates[w2] = 1
    if w1 in duplicates:
      continue
    unique_items.add(w1)
  print('Unique locations list:\n {}'.format(unique_items))
  return unique_items


if __name__ == "__main__":
  # Company Name Normalization
  cname_raw_data = ['Nividia India', 'Nvidia Ireland', 'Marks and Spencers Ltd', 'M&S Limited', 'M and S Limited', 'Intel LLC', 'INTEL ASIA']
  cname_data = pd.DataFrame(cname_raw_data, columns =['company_name'], dtype = 'string') 
  companyNameNormalisation(cname_data)

  # Serial Number Normalization
  serial_raw_data = ['XYZ 13423 / ILD', 'ABC/ICL/20891NC','XYZ 13423/ILD', 'xyz13423/ILD','abc/icl-20891NC']
  serial_data = pd.DataFrame(serial_raw_data, columns =['serial_number'], dtype = 'string') 
  serialNumberNormalisation(serial_data)

  # Physical Goods Normalization  
  pgoods_raw_data = ['hardwood table', 'wood', 'table', 'toy', 'lego', 'desk']
  pgoods_data = pd.DataFrame(pgoods_raw_data, columns =['physical_good'], dtype = 'string') 
  physicalGoodsNormalisation(pgoods_data, threshold=0.3)

  # Location Normalization
  loc_raw_data = ['LONDON', 'HONG KONG', 'ASIA', 'london, England, UK']
  loc_data = pd.DataFrame(loc_raw_data, columns =['location'], dtype = 'string') 
  locationNormalisation(loc_data, threshold=0.6)
