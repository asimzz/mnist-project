import torch
from sklearn.metrics import accuracy_score

def train_model1(train_data, test_data, classifier, optimizer, criterion,epochs=500):
  for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classifier(train_data.data)
    loss = criterion(train_data.targets, outputs)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
      print(f"Epoch [{epoch}], train_loss: {loss.item():.4f}")

  # Evaluation
  with torch.no_grad():
    classifier.eval()
    test_outputs = classifier(test_data.data)
    print
    _, y_pred_tensor = torch.max(test_outputs, 1)
    y_pred = y_pred_tensor.numpy()

  # Calculate accuracy

  accuracy = accuracy_score(torch.argmax(test_data.targets, dim=1), y_pred)
  print(f"Accuracy: {accuracy:.2f}")
  
  
def train_model(X_train, y_train, X_test, y_test, optimizer, criterion, model, epochs=500):
  for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(y_train, outputs)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f"Epoch [{epoch}], train_loss: {loss.item():.4f}")

  # Evaluation
  with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    print
    _, y_pred_tensor = torch.max(test_outputs, 1)
    y_pred = y_pred_tensor.numpy()

  # Calculate accuracy

  accuracy = accuracy_score(torch.argmax(y_test, dim=1), y_pred)
  print(f"Accuracy: {accuracy:.2f}")