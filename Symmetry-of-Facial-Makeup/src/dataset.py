from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
from utils import creat_logger

logger = creat_logger("dataset")


class SiameseNetworkDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "data",
        transform=None,
        type: str = "train",
        component="all",
    ):
        self.image_folder_dir = os.path.join(os.getenv("DATASET_DIR"), dataset_name)
        if type == "train":
            self.image_folder_dir = os.path.join(self.image_folder_dir, "train_data")
        elif type == "test":
            self.image_folder_dir = os.path.join(self.image_folder_dir, "test_data")
        elif type == "val":
            self.image_folder_dir = os.path.join(self.image_folder_dir, "val_data")
        if not os.path.isdir(self.image_folder_dir):
            logger.error(
                f"The provided path '{self.image_folder_dir}' is not a valid directory."
            )
            raise ValueError(
                f"The provided path '{self.image_folder_dir}' is not a valid directory."
            )
        self.transform = transform
        self.pimages = []
        self.component = component
        self._load_data()

    def _load_data(self):
        logger.info(f"{'-' * 50} DATA LOADING {'-' * 50}")
        for label in os.listdir(self.image_folder_dir):
            logger.info(f"Folder {label}:")
            lb = False if label == "label_0" else True
            image_folder_dir_lb = os.path.join(self.image_folder_dir, label)
            for comp in os.listdir(image_folder_dir_lb):
                if comp == self.component or self.component == "all":
                    comp_images_dir = os.path.join(image_folder_dir_lb, comp)
                    cnt = 0
                    for pair_image_dir in tqdm(
                        os.listdir(comp_images_dir),
                        desc=f"Loading {comp} data in folder {label}",
                        leave=False,
                    ):
                        try:
                            pimage_path = os.path.join(comp_images_dir, pair_image_dir)
                            if os.path.exists(pimage_path):
                                pimage = os.listdir(pimage_path)
                                image1_path = os.path.join(pimage_path, pimage[0])
                                image2_path = os.path.join(pimage_path, pimage[1])
                                if os.path.exists(image1_path) and os.path.exists(
                                    image2_path
                                ):
                                    self.pimages.append(
                                        {
                                            "image1": image1_path,
                                            "image2": image2_path,
                                            "label": lb,
                                        }
                                    )
                                    cnt += 1
                        except Exception as e:
                            logger.debug(f"Invalid image folder at: {pimage_path}")
                            logger.info(f"Error: {e}")
                    logger.debug(f"{comp} data: {cnt} pairs of images.")
                    logger.debug(f"The {comp} data is saved at {comp_images_dir}")
                    logger.info(f"{comp} data loading completed.")

        logger.info("Data loading completed.")
        logger.info(f"Total pairs of images: {len(self.pimages)}")
        logger.info(f"{'-' * 50} DATA LOADED {'-' * 50}\n\n")

    def __getitem__(self, index):
        item = self.pimages[index]
        image1_path = item["image1"]
        image2_path = item["image2"]
        label = item["label"]
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2, label)

    def __len__(self):
        return len(self.pimages)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((2, 2)), transforms.ToTensor()])
    image_folder_dir = os.getenv("DATASET_DIR")
    my_dataset = SiameseNetworkDataset(
        dataset_name="data0103", transform=transform, type="test"
    )
