import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageFolder(Dataset):
    def __init__(self, root_dir):
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = ImageFolder(root_dir="train_data/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


def test_image_gen():  # to generate lowers images for testing
    dataset = ImageFolder(root_dir="test_HR/")
    output_folder = "test_data_LR/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for idx in range(len(dataset)):
        low_res, _ = dataset[idx]
        low_res_pil = transforms.ToPILImage()(low_res.cpu().detach())
        img_file, _ = dataset.data[idx]
        low_res_pil.save(os.path.join(output_folder, f"lowres_{img_file}"))


if __name__ == "__main__":
    test()
