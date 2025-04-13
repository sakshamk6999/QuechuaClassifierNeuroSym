import torch
import numpy as np
import pandas as pd

class Regularization_Layer:
  def __init__(self):
    pass

  def regularize_logits(self, logits, reg_whole):
    total_score = torch.zeros_like(logits).to(device)

    for d in node2index.keys():
      if child2parent[d] is None:
        continue

      curr_logits = torch.zeros_like(logits).to(device)

      curr_logits[:, node2index[d]] = 1

      probab_node = logits * curr_logits

      parent_logits = torch.zeros_like(logits).to(device)

      parent_logits[:, node2index[child2parent[d]]] = 1

      probab_parent = logits * parent_logits

      #add reg here
      score = 1 - torch.minimum(1 - logits[:, node2index[d]] + logits[:, node2index[child2parent[d]]], torch.ones_like(logits[:, node2index[d]]))

      response_logits = torch.zeros_like(logits).to(device)
      response_logits[:, node2index[d]] = score

      total_score += response_logits

    total_score = total_score * reg_whole

    clipped_rules = torch.clamp(total_score, -60, 60)
    # print("clipped_rules", clipped_rules)
    return torch.exp(-1*clipped_rules)
