import numpy as np
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn

# Giả sử batch_size = 3
# output = torch.tensor([[1.0, 2.0],
#                        [3.0, 1.0],
#                        [0.5, 1.5]])
def caculate_metrics(label: torch.Tensor, output: torch.Tensor) -> tuple[float, float, float, float]:

  """
  Hàm tính toán các metrics đánh giá hiệu suất của mô hình.

  Args:
    label: Tensor chứa nhãn thật (ground truth).
    output: Tensor chứa đầu ra dự đoán của mô hình (logits).

  Returns:
    acc, precison, recall, f1
  """

  prob = torch.softmax(output, dim=1)
  pred = torch.argmax(prob, dim=1)

  # Chuyển sang numpy để sử dụng các hàm trong thư viện sklearn
  label = label.detach().cpu().numpy()  # Tách khỏi computational graph, chuyển lên CPU và sang numpy
  pred = pred.detach().cpu().numpy()   # Tách khỏi computational graph, chuyển lên CPU và sang numpy

  #-------------------Accuracy-----------------------------------------------

  acc = metrics.accuracy_score(label, pred)  # Tính độ chính xác

  #------------------Precision, Recall, F1-score--------------------------

  # average = "macro": Tính trung bình precision, recall, f1-score cho từng lớp (không tính trọng số)
  # zero_division=0: Nếu có lớp nào không có dự đoán đúng (chia cho 0), gán giá trị 0 thay vì báo lỗi.
  precison = metrics.precision_score(label, pred, average="macro", zero_division=0)
  recall = metrics.recall_score(label, pred, average="macro", zero_division=0)
  f1 = metrics.f1_score(label, pred, average="macro", zero_division=0)

  return acc, precison, recall, f1  # Trả về các giá trị metrics

class Recorder:
  def __init__(self):
    self.num = 0
    self.sum = 0
  def update(self, value):
    if value is not None:
      self.sum += value
      self.num += 1
  def average(self):
    if self.num == 0:
      return None
    return self.sum / self.num
  def clear(self):
    self.sum = 0
    self.num = 0