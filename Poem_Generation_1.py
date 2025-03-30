#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import import_ipynb
from Model import GPT
import pickle


# In[2]:


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(
    vocab_size=77610, 
    embed_dim=256, 
    seq_len=200, 
    dropout=0.1, 
    n_heads=8, 
    device=device
).to(device)
model.load_state_dict(torch.load("pretrained_model.pth", map_location=device))
model.eval()


# In[ ]:





# In[3]:


import torch.nn.functional as F

def sample_with_temperature(probabilities, temperature=1.7, top_k=10):
    """Apply temperature sampling and top-k filtering to add randomness."""
    probabilities = probabilities / temperature  # Adjust probability distribution
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    top_k_probs = sorted_probs[:top_k]
    top_k_indices = sorted_indices[:top_k]

    # Sample from the top-k tokens
    sampled_index = torch.multinomial(F.softmax(top_k_probs, dim=-1), 1).item()
    return top_k_indices[sampled_index].item()
def generate_text(prompt, max_length=50,temperature=1.7, top_k=10):
    # Convert prompt text to token sequence
    tokens = tokenizer.texts_to_sequences([prompt])[0]
    
    # Pad sequence to match model input length
    input_ids = pad_sequences([tokens], maxlen=200, padding="post", truncating="pre")
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)

    # Store generated tokens separately
    if not tokens:
        print("⚠️ Warning: Tokenizer returned an empty sequence. Ensure it was trained on your dataset.")
        return ""
    generated_tokens = tokens.copy()
    print("\n🔹 Initial Input IDs:", input_ids.tolist())

    with torch.no_grad():
        for step in range(200-len(tokens)):
            output = model(input_ids)  
            probabilities = torch.nn.functional.softmax(output[:, -1, :], dim=-1).squeeze()
            #print(f"Shape of probabilities: {probabilities.shape}")  
            next_token_id = sample_with_temperature(probabilities, temperature=temperature,top_k= top_k)

            
            generated_tokens.append(next_token_id)  # Append to generated sequence
            
            # Update input_ids by shifting left and adding the new token
            input_ids[0,step+len(tokens)]=next_token_id
            #input_ids = torch.cat(
                #[input_ids[:, 1:], torch.tensor([[next_token_id]], device=device)], dim=-1
            #)
            #print(f"\n🔹 Iteration {step+1}: Next Token ID = {next_token_id}")
            #print("   Updated Input IDs:", input_ids.tolist())
            # Stop if EOS token is generated
            if next_token_id == tokenizer.word_index.get("<eos>", None):  # Modify based on your dataset
                break

    generated_text = tokenizer.sequences_to_texts([generated_tokens])[0]
    
    print("\n🔹 Generated Text:\n", generated_text)  # Ensure text is printed
    return generated_text


# Example Usage


# In[4]:


generated_output = generate_text("Once upon a time")
print("\n✅ Final Output:\n", generated_output)


# In[5]:


generated_output = generate_text("I love water")
print("\n✅ Final Output:\n", generated_output)

