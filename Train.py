import torch
from QuechuaDataset import QuechuaDataSet
from torch.utils.data import DataLoader, random_split
from Model import CustomLoss, QuechuaClassifierStudent
from Preprocessing import node2index, get_hierarchy
from ScoreEval import evaluateF1

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def training(training_data_size, train_data, test_data, device):
  log_file = open("logs/logs.txt", "a")
  log_file.write(f"Beginning Model with Training size:{training_data_size}\n\n")
  train_dataset, _ = random_split(train_data, [training_data_size, len(train_data) - training_data_size])

  train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

  customLoss = CustomLoss(device).to(device)
  model = QuechuaClassifierStudent(device).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

  train_loss_history = []
  eval_f1_history = []

  for epoch in range(5):
    print(f"Epoch: {epoch}")
    log_file.write(f"---- EPOCH: {epoch}")
    model.train()
    for i, data in enumerate(train_data_loader, 0):
      inputs = data

      input_ids = inputs['input_ids'].to(device)

      attention_mask = inputs['attention_mask'].to(device)
      labels = inputs['targetLabels'].to(device)

      optimizer.zero_grad()

      outputs = model(input_ids.squeeze(1), attention_mask.squeeze(1)).to(device)

      loss = customLoss(outputs.squeeze(0), labels.squeeze(0).float(), i % 10 == 0)
      

      loss.backward()
      optimizer.step()

      if i % 100 == 0:
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        train_loss_history.append(loss.item())
        log_file.write(f"------ Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    # print(evaluateF1(model, test_data, device))
    eval_result = evaluateF1(model, test_data, device)
    eval_f1_history.append(eval_result)

    print("epoch", epoch, "overall_f1", eval_result['overall'])
    print("epoch", epoch, "dialect_f1", eval_result['dialect'])

    log_file.write(f"epoch: {epoch} overall_f1: {eval_result['overall']}\n")
    log_file.write(f"epoch: {epoch} dialect_f1: {eval_result['dialect']}\n")
  
  log_file.close()

  return model, train_loss_history, eval_f1_history

def main():

  USE_CUDA = torch.cuda.is_available()
  device = torch.device("cuda" if USE_CUDA else "cpu")

  child2parent = get_hierarchy()

  dataset = QuechuaDataSet('chunked_data.csv', node2index, child2parent)

  train_size = int(0.8 * len(dataset))

  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size

  # Use random_split to create the subsets
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  for i in [0.2, 0.4, 0.6, 0.8, 1]:
    print(f"Training with {i} samples")
    model, train_loss_history, eval_f1_history = training(int(i * len(train_dataset)), train_dataset, test_dataset, device)
    checkpoint(model, "checkpoints/model-" + str(i) +".pth")
  
if __name__ == "__main__":
    main()