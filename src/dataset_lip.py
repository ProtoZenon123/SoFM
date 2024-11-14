from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm


class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder_dir, transform=None):
        self.image_folder_dir = image_folder_dir
        self.transform = transform
        self.pimages = []
        self._load_data()
    
    def _load_data(self):
        for label in os.listdir(self.image_folder_dir):
            print(f"Folder {label}:")
            lb = True
            if label == 'label_0':
                lb = False
            image_folder_dir_lb = os.path.join(self.image_folder_dir, label)
            for comp in os.listdir(image_folder_dir_lb):
                if comp == 'lip':
                    comp_images_dir = os.path.join(image_folder_dir_lb, comp)
                    cnt = 0
                    for pair_image_dir in tqdm(os.listdir(comp_images_dir),desc=f"Loading {comp} data in folder {label}", leave=False):
                        try:
                            pimage_path = os.path.join(comp_images_dir, pair_image_dir)
                            pimage = os.listdir(pimage_path)
                            self.pimages.append({
                                'image1': os.path.join(pimage_path, pimage[0]),
                                'image2': os.path.join(pimage_path, pimage[1]),
                                'label': lb
                            })
                            cnt += 1
                        except:
                            print(f"Invalid image folder at: {pimage_path}")
                    print(f"{comp} data: {cnt} pairs of images.")
            
        print(f"Data loading completed.")
        print(f"Total pairs of images: {len(self.pimages)}")
        print("------------------------------------------------- DATA LOADED -------------------------------------------------\n\n")

    def __getitem__(self, index):
        item = self.pimages[index]
        image1_path = item['image1']
        image2_path = item['image2']
        label = item['label']
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2, label)

    def __len__(self):
        return len(self.pimages)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((2, 2)),
                                    transforms.ToTensor()
                                    ])
    image_folder_dir = '/media/nguyenminhquan/fshit/SymmetricalMakeup/data/train_data'
    my_dataset = SiameseNetworkDataset(image_folder_dir=image_folder_dir,
                                       transform=transform)
    
    for item in my_dataset:
        print(item)
    
