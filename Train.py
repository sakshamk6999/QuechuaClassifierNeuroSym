train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Use random_split to create the subsets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# bceLoss = nn.CrossEntropyLoss().to(device)
customLoss = CustomLoss().to(device)
model = QuechuaClassifierStudent().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

train_loss_history = []

for epoch in range(5):
  print(f"Epoch: {epoch}")
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
      print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

