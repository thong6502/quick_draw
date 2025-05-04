from src.model import QuickDrawModel
from src.dataset import QuickDrawDataset
import torch
import datetime
import torch.nn as nn
from torch.optim import adam
from src.config import *
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from src.base_metric_class import Recorder
from src.logger import create_logger
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args():
  parser = ArgumentParser()
  parser.add_argument("-r", "--root", type=str, default="data", help="root path dataset")
  parser.add_argument("-e", "--total_epochs", type=int, default=100, help="Total epoch")
  parser.add_argument("-tl", "--train_split", type=float, default=0.8, help="split for train")
  parser.add_argument("-b", "--batch-size", type=int, default=60, help="batch size on train")
  parser.add_argument("-lr", "--lr", type=float, default=0.001, help="learning rate on train")
  parser.add_argument("-ckp", "--checkpoint", type=str, default="checkpoint/last_checkpoint.pth")
  parser.add_argument("-mp", "--model-path", type=str, default="checkpoint/best_model.pth")
  args = parser.parse_args()
  return args


class Trainer(object):
  def __init__(self,
              model,
              optimizer,
              logger):
    self.model = model
    self.optimizer = optimizer
    self.logger = logger
    self.writer = self.get_writer()
    self.speed_up() #move model to GPU

  def createConfusionMatrix(self, y_true, y_pred):
      # Build confusion matrix
      cf_matrix = confusion_matrix(y_true, y_pred)
      df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in CLASSES],
                          columns=[i for i in CLASSES])
      plt.figure(figsize=(12, 7))    
      return sn.heatmap(df_cm, annot=True).get_figure()
  
  def get_writer(self, tensorboard_folder = Path("logs/tensorboard")):
    if not tensorboard_folder.exists():
      tensorboard_folder.parent.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(tensorboard_folder)
  
  def speed_up(self):
    self.model.to(device)
    self.model.device = device

  def set_train(self):
    self.model.train()
    self.train = True

  def set_eval(self):
    self.model.eval()
    self.train = False

  def load_model(self, model_path:str) -> None:
    if Path(model_path).is_file():
      saved = torch.load(model_path, map_location="cpu")
      self.model.load_state_dict(saved)
      self.logger.info(f"model found in {str(model_path)}")
    else:
      raise NotImplementedError(f"===> No model fount at {str(model_path)}")
  
  def save_model(self, model_path:str) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), model_path)
    self.logger.info(f"Model saved to {str(model_path)}")
  
  def save_ckpt(self, save_path:str, checkpoint:dict) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    self.logger.info(f"Check point saved to {str(save_path)}")

  def load_ckpt(self, path_ckpt:str) -> dict:
    path_ckpt = Path(path_ckpt)
    if path_ckpt.is_file():
      ckpt = torch.load(path_ckpt, map_location="cpu")
      return ckpt
    else:
      raise NotImplementedError(f"===> No checkpoint fount at {str(path_ckpt)}")

  def train_epoch(self, train_loader, epoch, total_epochs):

    postfix_dict = defaultdict(float)
    torch.backends.cudnn.benchmark = True
    self.set_train()
    #define training recorder
    train_recorder_loss = defaultdict(Recorder)
    train_recorder_metric = defaultdict(Recorder)

    num_iter_for_train = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Train epoch [{epoch + 1} / {total_epochs}]", colour="cyan")


    for iteration, (images, labels) in enumerate(pbar):

      # Sao chép dữ liệu sang gpu và thực hiện việc đó một cách bất đồng bộ
      # Để hiệu quả dữ liệu phải nằm trong vùng nhớ (pinned memory) pin_memory=True khi tạo DataLoader
      images = images.to(device, non_blocking = True)
      labels = labels.to(device, non_blocking = True)

      self.optimizer.zero_grad()

      outputs = self.model(images)
      losses = self.model.get_loss(labels, outputs)
      
      losses["overall"].backward()
      self.optimizer.step()

      metric_batch_dict = self.model.get_metrics(labels, outputs)

      for name, value in losses.items():
        train_recorder_loss[name].update(value.item())
        postfix_dict[name] = train_recorder_loss[name].average()
      for name, value in metric_batch_dict.items():
        train_recorder_metric[name].update(value)
        postfix_dict[name] = train_recorder_metric[name].average()

      pbar.set_postfix(postfix_dict)

      global_step = epoch * num_iter_for_train + iteration
      if global_step == 0:
        continue
      if global_step % 300 == 0:
        if any(value == 0 for value in postfix_dict.values()):
          continue
        for k, v in train_recorder_loss.items():
          self.writer.add_scalar(f"Train/{k}",v.average(), global_step)

        for k,v in train_recorder_metric.items():
          self.writer.add_scalar(f"Train/{k}", v.average(), global_step)

        
        #clear recorder
        for v in train_recorder_loss.values():
          v.clear()
        for v in train_recorder_metric.values():
          v.clear()
  
  def val_epoch(self, val_loader, epoch, total_epochs):
    self.set_eval()
    y_pred = [] # save predction
    y_true = [] # save ground truth
    postfix_dict = defaultdict(float)
    metric_dict = defaultdict(float)
    pbar = tqdm(val_loader, desc=f"Val epoch [{epoch + 1} / {total_epochs}]")

    recorder_metric = defaultdict(Recorder)
    recorder_loss = defaultdict(Recorder)
    for iteration, (images, labels) in enumerate(pbar):
      images = images.to(device, non_blocking = True)
      labels = labels.to(device, non_blocking = True)

      with torch.no_grad():
        outputs = self.model(images)
        losses = self.model.get_loss(labels, outputs)


      metric_batch_dict = self.model.get_metrics(labels, outputs)

      preds = torch.argmax(outputs, dim=1).cpu().numpy()
      y_pred.extend(preds)

      labels = labels.data.cpu().numpy()
      y_true.extend(labels)

      for name, value in losses.items():
        recorder_loss[name].update(value.item())
        postfix_dict[name] = recorder_loss[name].average()
      for name, value in metric_batch_dict.items():
        recorder_metric[name].update(value)
        postfix_dict[name] = recorder_metric[name].average()
      
      pbar.set_postfix(postfix_dict)

    for k, v in recorder_loss.items():
      self.writer.add_scalar(f"Val/{k}", v.average(), epoch)
      metric_dict[k] = v.average()
    for k, v in recorder_metric.items():
      self.writer.add_scalar(f"Val/{k}", v.average(), epoch)
      metric_dict[k] = v.average()

    self.writer.add_figure("Confusion matrix", self.createConfusionMatrix(y_true, y_pred), epoch)
    return metric_dict

def main(args):
  total_epochs = args.total_epochs
  num_worker = multiprocessing.cpu_count()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  model = QuickDrawModel(len(CLASSES))
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
  
  time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  logger_path = f"logs/training/quickdraw_{time_now}/trainning.log"
  logger = create_logger(logger_path)
  logger.info(f"save log to '{logger_path}'")

  dataset = QuickDrawDataset(args.root)
  train_set, val_set = random_split(dataset, [args.train_split, 1 - args.train_split])
  train_loader = DataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_worker,
                            pin_memory=True)
  val_loader = DataLoader(val_set,
                          batch_size=args.batch_size,
                          num_workers=num_worker,
                          pin_memory=True)
  

  trainer = Trainer(model, optimizer=optimizer, logger=logger)

  if Path(args.checkpoint).is_file():
    ckpt = trainer.load_ckpt(Path(args.checkpoint))
    start_epoch = ckpt["epoch"]
    best_loss = ckpt["best_loss"]
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
  else:
    start_epoch = 0
    best_loss = 999

  for epoch in range(start_epoch, total_epochs):
    logger.info(f"===> epoch[{epoch}/{total_epochs}] start")
    trainer.train_epoch(train_loader, epoch, total_epochs)
    metric_dict = trainer.val_epoch(val_loader, epoch, total_epochs)

    scheduler.step()
    checkpoint = {
      "epoch": epoch + 1,
      "model_state_dict": trainer.model.state_dict(),
      "optimizer_state_dict": trainer.optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict(),
      "best_loss":best_loss,
      "training_metric": metric_dict
    }

    last_checkpoint_path = args.checkpoint
    best_checkpoint_path = args.checkpoint.replace("last","best")
    if best_loss > metric_dict["overall"]:
      checkpoint["best_loss"] = metric_dict["overall"]
      trainer.save_ckpt(save_path=best_checkpoint_path,checkpoint=checkpoint)
      trainer.save_model(args.model_path)
    
    trainer.save_ckpt(save_path=last_checkpoint_path,checkpoint=checkpoint)



if __name__ == "__main__":
  args = get_args()
  main(args)