from pathlib import Path
import logging
import datetime
import torch.distributed as dist

class RankFilter(logging.Filter):
  def __init__(self, rank):
    self.rank = rank

  def filter(self, record):
    return dist.get_rank() == self.rank
  
def create_logger(log_path:str) -> None:
  #Create log path
  log_path = Path(log_path)
  log_path.parent.mkdir(parents=True, exist_ok=True)

  #create logger object
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  #Create file handler and set the formatter
  fh = logging.FileHandler(log_path)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - [in %(pathname)s:%(lineno)d] %(message)s')
  fh.setFormatter(formatter)

  # Add the file handler to the logger
  logger.addHandler(fh)

  # Add a stream handler to print console
  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO) # Set logging level for stream handler
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger

if __name__ == "__main__":
  log_dir = "./logs/training/"
  model_name = "quickdraw"
  timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  logger = create_logger( Path(log_dir) / f"{model_name}_{timenow}" / "training.log" )
  logger.info("this is massage")