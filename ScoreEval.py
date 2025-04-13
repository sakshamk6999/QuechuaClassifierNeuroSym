from sklearn.metrics import f1_score
import torch
import numpy as np

def evaluateF1Samples(model, dataset, label):
  model.eval()

  predicitons = []
  actual_np = []

  with torch.no_grad():
    for i in range(len(dataset)):
      prediction = model(dataset[i]['input_ids'].to(device), dataset[i]['attention_mask'].to(device))
      predicitons.append(prediction)
      actual_np.append(dataset[i][label].numpy())

    print(predicitons)
    predcition_tensor = torch.sigmoid(torch.stack(predicitons).squeeze(1))
    predicted_labels_np = torch.where(predcition_tensor > 0.5, 1, 0).cpu().numpy()
    actual_np = np.array(actual_np)

    print("prediction", predicted_labels_np.shape, "actual", actual_np.shape)

    f1_avg_score_samples = f1_score(actual_np, predicted_labels_np, average='samples')

    return f1_avg_score_samples