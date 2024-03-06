import glob
import pathlib
from sklearn.model_selection import train_test_split
import util as ut
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma

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

def apply_random_noise(img, SNRs=[10, 15, 20, 25]):
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

def NLM_denoising(img):
    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    patch_kw = dict(patch_size=5,
                    patch_distance=6,
                    channel_axis=-1)
    denoise2_fast = denoise_nl_means(np.expand_dims(img, axis=2), h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    return denoise2_fast

def load_images(args):
    dir_img = pathlib.Path(args.pathData)
    files = glob.glob(str(dir_img/'*.jpg'))
    if args.include_orig:
        images = torch.empty(len(files), 3, 224, 224)
    else:
        images = torch.empty(len(files), 2, 224, 224)
    one_hot_labels, addInfos = torch.empty(len(files), 5), torch.empty(len(files), 4)
    for i, file in enumerate(tqdm(files)):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        if args.denoise_data:
            img = NLM_denoising(img)
        img_b, label = apply_random_noise(img)
        addInfo = get_additional_image_info(img_b)
        _, _, _, cD = ut.calculate_dwt(img_b)
        img_energy = ut.calculate_energy(img_b)
        img_energy, cD = torch.from_numpy(np.expand_dims(img_energy, 0)), torch.from_numpy(np.expand_dims(cv2.resize(cD, (224, 224)), 0))
        if args.include_orig:
            images[i] = torch.cat((torch.from_numpy(np.expand_dims(img_b, 0)), img_energy, cD), dim=0)
        else:
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



    


