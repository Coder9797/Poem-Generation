#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[2]:


import os


# In[3]:


import zipfile

zip_path = "forms.zip"
corpus = []

print("Loading poems in corpus...\n")

with zipfile.ZipFile(zip_path, "r") as z:
    for filename in z.namelist():
        if filename.endswith(".txt"):  # Ensure it's a text file
            print(f"Processing: {filename}")  # Print filename before reading
            with z.open(filename) as file:
                corpus.extend(file.read().decode("utf-8").split("\n"))


# In[4]:


len(corpus)


# In[5]:


with open("Poems.txt", "w") as file:
    for line in corpus:
        file.write(line + "\n")


# In[6]:


corpus[:20]


# In[7]:


import string

def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

corpus = [ remove_punc(s.lower().strip()) for s in corpus ]


# In[8]:


corpus[:20]


# In[9]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[10]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)


# In[11]:


vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")


# In[12]:


n_grams = []
max_sequence_len = 0

for sentence in corpus:
    # convert sentence to tokens
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(2, len(tokens)+1):
        # extract n-gram
        n_gram = tokens[:i]
        # save n-gram
        n_grams.append(n_gram)
        # calculate maximum sequence length
        if len(n_gram) > max_sequence_len:
            max_sequence_len = len(n_gram)
        
print(f"Number of n-grams: {len(n_grams)}")
print(f"Maximum n-gram length: {max_sequence_len}")


# In[13]:


for n_gram in n_grams[:20]:
    print(n_gram)


# In[14]:


padded_n_grams = np.array(pad_sequences(n_grams, maxlen=200, padding="post", truncating="pre"))

padded_n_grams.shape


# In[15]:


for seq in padded_n_grams[:3]:
    print(seq)


# In[16]:


X = padded_n_grams[:, :-1]
y = padded_n_grams[:, -1]

print(f"X: {X.shape}")
print(f"y: {y.shape}")


# In[17]:


Y_tensor = torch.tensor(y)


# In[18]:


from torch.utils.data import TensorDataset, DataLoader
def create_lm_sequences(tokenized_padded_text, seq_length):
    # Convert to numpy array if not already
    tokens = np.array(tokenized_padded_text)
    
    # Create X by taking all tokens except last
    x = tokens[:-1]
    
    # Create Y by taking all tokens except first
    y = tokens[1:]
    
    return x, y


x1, y1 = create_lm_sequences(padded_n_grams, seq_length=6)


# In[19]:


X1=torch.tensor(x1)
Y1=torch.tensor(y1)
X1=X1.long()
Y1=Y1.long()


# In[20]:


dataset = TensorDataset(X1, Y1)


# In[21]:


batch_size = 32  # Adjust based on GPU memory and model size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[ ]:




