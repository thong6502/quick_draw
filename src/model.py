import torch
import torch.nn as nn
from .base_metric_class import caculate_metrics

class QuickDrawModel(nn.Module):
  def __init__(self, num_classes = 15, input_size = 28):
    super().__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 32, 3, padding="same"),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2)
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 64, 3, padding="same"),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2)
    )
    dimension = int(64 * (input_size / 4) ** 2)
    self.fc1 = nn.Sequential(nn.Dropout(inplace=True), nn.Linear(dimension, 512))
    self.fc2 = nn.Sequential(nn.Dropout(inplace=True), nn.Linear(512, 124))
    self.fc3 = nn.Sequential(nn.Dropout(inplace=True), nn.Linear(124, num_classes))

  def get_metrics(self, labels: torch.Tensor, outputs:torch.Tensor) -> dict:
    acc, precision, recall, f1 = caculate_metrics(labels, outputs)
    metric_batch_dict = {"acc":acc,
                          "precision": precision,
                          "recall": recall,
                          "f1": f1}
    return metric_batch_dict

  def get_loss(self, labels:torch.Tensor, outputs:torch.Tensor) -> dict:
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    losses = {"overall": loss, "loss": loss}
    return losses

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
  
if __name__ == "__main__":
  x = torch.randn((5,1,28,28))
  model = QuickDrawModel()
  x = model(x)
  print(x.size())