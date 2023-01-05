# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

#Spacy
import spacy
nlp = spacy.load('en_core_web_sm')

# Other
import re
import json
import string
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#Keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

#SKlearn
from sklearn.preprocessing import LabelEncoder