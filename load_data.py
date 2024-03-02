import glob
import pathlib
from sklearn.model_selection import train_test_split
import util as ut
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
  def __init__(self, x_data, y_labels, z_snr):
        self.x = x_data
        self.y = y_labels
        self.z = z_snr

  def __len__(self):
        return len(self.x)

  def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]
        z = self.z[index]
        return X, y, z

def apply_random_noise(img, SNRs=[10, 15, 20]):
    noises = ["gaussian", "speckle", "salt_and_pepper", "poisson", "None"]
    noise_choice = random.choice(noises)
    SNR_choice = random.choice(SNRs)
    match noise_choice:
        case "None":
            return img, torch.tensor([1, 0, 0, 0, 0])
        case "gaussian":
            img = ut.gaussian_noise(SNR_choice, img)
            return img, torch.tensor([0, 1, 0, 0, 0])
        case "speckle":
            img = ut.speckle_noise(SNR_choice, img)
            return img, torch.tensor([0, 0, 1, 0, 0])
        case "salt_and_pepper":
            img = ut.salt_and_pepper(SNR_choice, img)
            return img, torch.tensor([0, 0, 0, 1, 0])
        case "poisson":
            img = ut.poisson_noise(SNR_choice, img)
            return img, torch.tensor([0, 0, 0, 0, 1])
        case _:
            raise ValueError("Erreur random noise choice")

def get_additional_image_info(img):
    kurtosis_val, skewness_val, entropy_val, correlation_val = ut.calculate_kurtosis(img), ut.calculate_skewness(img), ut.calculate_entropy(img), ut.compute_correlation(img)
    return torch.tensor([kurtosis_val, skewness_val, entropy_val, correlation_val])

def load_images(args):
    dir_img = pathlib.Path(args.pathData)
    files = glob.glob(str(dir_img/'*.jpg'))
    images, one_hot_labels, addInfos = torch.empty(len(files), 2, 224, 224), torch.empty(len(files), 5), torch.empty(len(files), 4)
    for i, file in enumerate(tqdm(files)):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        img, label = apply_random_noise(img)
        addInfo = get_additional_image_info(img)
        _, _, _, cD = ut.calculate_dwt(img)
        img_energy = ut.calculate_energy(img)
        img_energy, cD = torch.from_numpy(np.expand_dims(img_energy, 0)), torch.from_numpy(np.expand_dims(cv2.resize(cD, (224, 224)), 0))
        images[i] = torch.cat((img_energy, cD), dim=0)
        one_hot_labels[i] = label
        addInfos[i] = addInfo
    return images, one_hot_labels, addInfos

def load_dataset(args):
    X, Y, Z = load_images(args)
    if args.visualise_data:
        ut.visualise_data(args.pathSave, Y, Z)
    x_train, x_test, y_train, y_test, z_train, z_test, = train_test_split(X, Y, Z, test_size=args.testSize, shuffle=args.shuffle)
    dataset_train = Dataset(x_train, y_train, z_train)
    dataset_test = Dataset(x_test, y_test, z_test)
    kwargs = {'num_workers': args.numWorkers, 'pin_memory': True} if args.device=='cuda' else {}
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    return train_loader, test_loader



    


