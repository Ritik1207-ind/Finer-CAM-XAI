from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CUB200Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.transform = transform

        images_file = self.root / "images.txt"
        labels_file = self.root / "image_class_labels.txt"
        split_file = self.root / "train_test_split.txt"

        images = {}
        labels = {}
        splits = {}

        with open(images_file, "r") as f:
            for line in f:
                image_id, image_path = line.strip().split(maxsplit=1)
                images[int(image_id)] = image_path

        with open(labels_file, "r") as f:
            for line in f:
                image_id, label = line.strip().split()
                labels[int(image_id)] = int(label) - 1

        with open(split_file, "r") as f:
            for line in f:
                image_id, flag = line.strip().split()
                splits[int(image_id)] = int(flag)

        self.samples = []

        for image_id in sorted(images.keys()):
            is_train = splits[image_id] == 1

            if is_train == train:
                self.samples.append(
                    (
                        image_id,
                        self.root / "images" / images[image_id],
                        labels[image_id],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def get_image_id(self, index):
        return self.samples[index][0]

    def get_image_path(self, index):
        return self.samples[index][1]

    def __getitem__(self, index):
        image_id, image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
        ##
