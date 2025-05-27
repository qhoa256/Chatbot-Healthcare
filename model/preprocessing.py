import re
import random
import pandas as pd
import numpy as np
from pyvi import ViTokenizer
from collections import Counter
import string, unicodedata
from .utils import clean_data
pd.set_option('mode.chained_assignment', None)

data= pd.read_excel('data/dataset.xlsx', 'true_value')
data.columns = ['Questions',	'Full_questions',	'Answers']
data = data[data['Answers'].str.len() < 550]

input_docs = []
output_docs = []
input_tokens = set()
output_tokens = set()

pairs=[]

for i in range(data.shape[0]):
  pairs.append(((data['Questions'][i]),data['Answers'][i]))

for line in pairs:

  input_doc, output_doc = line[0], line[1]
  input_doc = clean_data(input_doc)
  input_docs.append(input_doc)

  output_doc = clean_data(output_doc)
  output_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", output_doc))

  output_doc = '<START> ' + output_doc + ' <END>'

  output_docs.append(output_doc)

  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    if token not in input_tokens:
      input_tokens.add(token)
  for token in output_doc.split():
    if token not in output_tokens:
      output_tokens.add(token)

input_tokens = sorted(list(input_tokens))
output_tokens = sorted(list(output_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(output_tokens)

input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(output_tokens)])
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", output_doc)) for output_doc in output_docs])

encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for line, (input_doc, output_doc) in enumerate(zip(input_docs, output_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(output_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.