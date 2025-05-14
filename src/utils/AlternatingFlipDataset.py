from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class AlternatingFlipDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        img, label = self.base_dataset[index]

        should_flip = self.epoch > 0 and (self.epoch + index) % 2 == 1
        if should_flip:
            img = F.hflip(img)

        return img, label

    def __len__(self):
        return len(self.base_dataset)