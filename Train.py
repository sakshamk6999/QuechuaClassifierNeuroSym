import torch
from QuechuaDataset import QuechuaDataSet
from torch.utils.data import DataLoader, random_split
from Model import CustomLoss, QuechuaClassifierStudent
from Preprocessing import node2index, get_hierarchy
from ScoreEval import evaluateF1

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def main():
  log_file = open("logs/logs.txt", "w")

  USE_CUDA = torch.cuda.is_available()
  device = torch.device("cuda" if USE_CUDA else "cpu")

  child2parent = get_hierarchy()

  dataset = QuechuaDataSet('chunked_data.csv', node2index, child2parent)

  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size

  # Use random_split to create the subsets
  train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])

  train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

  # bceLoss = nn.CrossEntropyLoss().to(device)
  customLoss = CustomLoss(device).to(device)
  model = QuechuaClassifierStudent(device).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

  train_loss_history = []
  eval_f1_history = []
  
  for epoch in range(5):
    print(f"Epoch: {epoch}")
    log_file.write(f"Epoch: {epoch}\n")
    for i, data in enumerate(train_data_loader, 0):
      inputs = data

      input_ids = inputs['input_ids'].to(device)

      attention_mask = inputs['attention_mask'].to(device)
      labels = inputs['targetLabels'].to(device)

      optimizer.zero_grad()

      # print("input shape", input_ids.shape, "attention", attention_mask.shape)

      outputs = model(input_ids.squeeze(1), attention_mask.squeeze(1)).to(device)

      loss = customLoss(outputs.squeeze(0), labels.squeeze(0).float(), i % 100 == 0)

      # print(f"grad_fn: {loss.grad_fn}") 

      train_loss_history.append(loss.item())

      loss.backward()
      optimizer.step()

      if i % 100 == 0:
        print(f"\tEpoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        log_file.write(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}\n")
  
    checkpoint(model, "checkpoints/model.pth")
    eval_result = evaluateF1(model, eval_dataset, device)
    eval_f1_history.append(eval_result)

    print("epoch", epoch, "overall_f1", eval_result['overall'])
    print("epoch", epoch, "dialect_f1", eval_result['dialect'])

    log_file.write(f"epoch: {epoch} overall_f1: {eval_result['overall']}\n")
    log_file.write(f"epoch: {epoch} dialect_f1: {eval_result['dialect']}\n")

  log_file.close()
  
if __name__ == "__main__":
    main()