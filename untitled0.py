# ECM2
# Case: Mansikka
# predicting strawberries sales in Finland
# Developer: Henry Bol


# =============================================================================
# Import the libraries
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import re

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize


# =============================================================================
# PART I: Load the data
# =============================================================================
# dataset_google_trends = pd.read_csv('google_mansikka.csv')
dataset_google_trends = pd.read_csv('google_mansikka_5years.csv')
df = dataset_google_trends
df.reindex(inplace=True)
