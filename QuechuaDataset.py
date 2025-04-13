from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QuechuaDataSet(Dataset):
  def __init__(self, csv_file, node2index, child2parent):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained("Llamacha/QuBERTa")
    self.data_df = pd.read_csv(csv_file)
    self.node2index = node2index
    self.child2parent = child2parent

  def __len__(self):
    return self.data_df.shape[0]

  def __getitem__(self, index):
    row = self.data_df.iloc[index]
    chunk = row['text']

    encoded_chunks = self.tokenizer(chunk, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    labels = torch.zeros(38, dtype=torch.long)

    q = row['dialect']
    while q is not None:
      labels[self.node2index[q]] = 1
      q = self.child2parent[q]

    return {
      **encoded_chunks,
      "dialect": row['dialect'],
      "targetLabels": labels
    }


