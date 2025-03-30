#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup  # Warm-up scheduler
import import_ipynb
from Model import GPT  # Import model
from Data_processing import dataloader  # Import tokenized data

import torch.optim as optim
# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(vocab_size=77610, embed_dim=256, seq_len=200, dropout=0.1,n_heads=8,device=device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs.view(-1, 77610), targets.view(-1))  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "pretrained_model.pth")
print("Model saved!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




