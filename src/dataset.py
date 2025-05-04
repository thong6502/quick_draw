# import cv2
from torch.utils.data import Dataset
from .config import CLASSES
import numpy as np
from pathlib import Path
from pprint import pprint
from random import shuffle


class QuickDrawDataset(Dataset):
  def __init__(self, root):
    super().__init__()
    self.root = Path(root)

    if not self.root.exists():
      raise FileNotFoundError(f"Directory not found: {self.root}")
    
    self.samples = []
    list_npy = self.root.glob("*.npy")
    for file_npy in list_npy:
      if file_npy.stem in CLASSES:
        label_idx = CLASSES.index(file_npy.stem)
        sample = np.load(file_npy)
        print(f"Found {len(sample)} {file_npy.stem} samples")
        self.samples.extend([(s, label_idx) for s in sample])

    print(f"Total {len(self.samples)} samples")
    shuffle(self.samples)
  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):
    img, label = self.samples[index]
    img = img.reshape(1, 28, 28).astype(np.float32) / 255.0
    # img = img.reshape(28,28,1)
    return img, label



if __name__ == "__main__":
  from pprint import pprint
  import numpy as np
  dataset = QuickDrawDataset("../data")
  img, label = dataset[0]

  # cv2.imwrite("image.jpg", img)

  # pprint(img.shape)
  # cv2.imshow("QuickDraw Image", img)
  # key = cv2.waitKey(5000)  # chờ 500ms (nửa giây)

  # cv2.destroyAllWindows()


