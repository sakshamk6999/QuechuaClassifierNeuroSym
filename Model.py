
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,  DataCollatorWithPadding, AutoModel
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, RandomSampler, Dataset, ConcatDataset, random_split
from torch.nn.utils.rnn import pad_sequence

class QuechuaClassifier(nn.Module):
    def __init__(self):
      super(QuechuaClassifier, self).__init__()
      self.quBert = AutoModel.from_pretrained("Llamacha/QuBERTa")
      self.classifier = nn.Linear(768, 38)

    def forward(self, input_ids, attention_mask):
      output = self.quBert(input_ids, attention_mask)
      output = output.last_hidden_state[:, 0, :]
      output = self.classifier(output)
      return output

class QuechuaClassifierStudent(nn.Module):
  def __init__(self):
    super(QuechuaClassifierStudent, self).__init__()
    self.quechuaClassifier = QuechuaClassifier()
    # self.classifier = nn.Linear(38, 38)
    self.rule_calculation = Regularization_Layer()

  def forward(self, input_ids, attention_mask):
    # print("input_ids shape", input_ids.shape, "attention_mask", attention_mask.shape)
    output = self.quechuaClassifier(input_ids, attention_mask)
    # print("output shape", output.shape)
    # print("output requires grad", output.requires_grad)
    regularized = self.rule_calculation.regularize_logits(output, 1000)
    regularized = output * regularized
    # print("regularized requires grad", regularized.requires_grad)

    combined = torch.cat((output, regularized), dim=1)
    # print("combined requires grad", combined.requires_grad)
    # output = output + rule_regularized_output
    # output = self.classifier(output)
    return combined
  
import torch.nn.functional as F

class CustomLoss(nn.Module):
  def __init__(self):
    super(CustomLoss, self).__init__()
    self.BCE = nn.BCEWithLogitsLoss()
    self.KLDiv = F.kl_div
    self.teacher_loss = 0
    self.student_loss = 0

  def get_loss(self):
    return self.teacher_loss, self.student_loss

  def forward(self, logits, labels, should_print):
    # print("logits require grad", logits.requires_grad)
    student_logits = logits[:, :38]
    student_logits_sigmoid = torch.sigmoid(student_logits)
    regularized_logits = torch.sigmoid(logits[:, 38:])

    bceLoss = self.BCE(student_logits, labels.float())
    totalKldLoss = torch.tensor(0.0).to(device)
    self.student_loss = bceLoss
    # print("student_logits_sigmoid", student_logits_sigmoid.requires_grad)
    # print("student_logits", student_logits.shape, "regularized_logits", regularized_logits.shape)
    for i in range(student_logits.shape[0]):
      for j in range(38):
        temp = self.KLDiv(torch.tensor([student_logits_sigmoid[i, j], 1 - student_logits_sigmoid[i, j]], requires_grad=True).log(), torch.tensor([regularized_logits[i, j], 1 - regularized_logits[i, j]], requires_grad=True).log(), reduction="batchmean", log_target=True)
        # print("temp", temp.requires_grad)
        totalKldLoss += temp

    totalKldLoss = totalKldLoss / (38 * student_logits.shape[0])

    self.teacher_loss = totalKldLoss

    if should_print:
      print("bceLoss", bceLoss, "kldLoss", totalKldLoss)

    return bceLoss + totalKldLoss
    # return totalKldLoss