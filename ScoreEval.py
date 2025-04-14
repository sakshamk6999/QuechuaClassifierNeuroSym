from sklearn.metrics import f1_score
import torch
import numpy as np

def evaluateF1(model, dataset, device):
    model.eval()

    all_predictions = []

    all_actual = []

    with torch.no_grad():
        for i in range(len(dataset)):
            # print("shapes are", dataset[i]['input_ids'].shape)
            outputs = model(dataset[i]['input_ids'].to(device), dataset[i]['attention_mask'].to(device))

            all_predictions.append(outputs[:,:38])
            all_actual.append(dataset[i]['targetLabels'].numpy())

    all_predictions_tensor = torch.sigmoid(torch.stack(all_predictions).squeeze(1))
    all_predictions_labels_np = torch.where(all_predictions_tensor > 0.5, 1, 0).cpu().numpy()
    all_actual_np = np.array(all_actual)

    # print(all_actual_np.shape, all_predictions_labels_np.shape)
    f1_score_all_micro = f1_score(all_actual_np, all_predictions_labels_np, average='micro')
    f1_score_all_macro = f1_score(all_actual_np, all_predictions_labels_np, average='macro')

    # print(all_actual_np[:, 14:].shape, all_predictions_labels_np[:, 14:].shape)
    f1_dialect_score_micro = f1_score(all_actual_np[:, 14:], all_predictions_labels_np[:, 14:], average='micro')
    f1_dialect_score_macro = f1_score(all_actual_np[:, 14:], all_predictions_labels_np[:, 14:], average='macro')

    return {
        "overall": [f1_score_all_macro, f1_score_all_micro],
        "dialect": [f1_dialect_score_macro, f1_dialect_score_micro]
    }


          
