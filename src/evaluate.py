import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import load_config
import model, dataset

def predict_labels(outputs, threshold=0.5):
    return (outputs >= threshold)

def test(test_dataloader, model, device, threshold):
    model.eval()

    num_correct_labels = 0
    total = 0
    threshold = threshold
    
    with torch.no_grad():
        for pimage in tqdm(test_dataloader, desc="Testing: "):
            image1 = pimage[0]
            image2 = pimage[1]
            label = pimage[2].float()
            label = label.view(len(label), -1)

            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            model = model.to(device)

            outputs = model(image1, image2)
            predict_label = predict_labels(outputs, threshold)

            total += len(label)
            num_correct_labels += (predict_label == label).sum().item()
            
    accuracy = num_correct_labels / total * 100
    return accuracy

def evaluate():
    config = load_config()
    
    test_batch_size = config['train_batch_size']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    train_dir = config['train_dir']
    # folder_weights_dir = config['folder_weights_dir']
    folder_weights_dir = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/WEIGHTS_ALL/weights_0511"
    
    base_dir = os.path.dirname(train_dir)
    test_dir = os.path.join(base_dir, 'test_data')
    test_dir = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data4train/data/test_data'
    
    transform = transforms.Compose([transforms.Resize((121, 121)),
                                    transforms.ToTensor()
                                    ])
    data = dataset.SiameseNetworkDataset(image_folder_dir=test_dir,
                                         transform=transform)

    dataloader = DataLoader(dataset=data,
                            batch_size=test_batch_size,
                            shuffle=True)
    cuda = 'cuda:' + f'{cuda_id}'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    
    model_last_weight = model.siameseNN(channels_img, num_features)
    model_min_loss_weight = model.siameseNN(channels_img, num_features)

    # last_weights_path = os.path.join(folder_weights_dir, 'last_weights.pth')
    last_weights_path = os.path.join(folder_weights_dir, 'last_weights.pth_4')
    min_loss_weight_path = os.path.join(folder_weights_dir, 'last_weights.pth_5')
    
    model_last_weight.load_state_dict(torch.load(last_weights_path))
    model_min_loss_weight.load_state_dict(torch.load(min_loss_weight_path))
    
    accuracy_01 = test(dataloader, model_last_weight, device, threshold)
    print(f"Accuracy with final weight: {accuracy_01:.2f}%")
    accuracy_02 = test(dataloader, model_min_loss_weight, device, threshold)
    print(f"Accuracy with min_loss weight: {accuracy_02:.2f}%")

if __name__ == '__main__':
    print("Evaluate with two model_weights:")
    print("\n\tNOTE: THESE WEIGHTS IS 4TH and 5TH WEIGHTS!!!\n")
    evaluate()
    