import pywt
from scipy.stats import skew
from scipy.stats import kurtosis
from skimage.measure import shannon_entropy
from scipy import ndimage
import numpy as np
import skimage as skimage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

def gaussian_noise(value, image):
    return skimage.util.random_noise(image.copy(), mode='gaussian', var = gaussian_db(value, image.shape[0]*image.shape[1], image.copy()), clip=True)

def gaussian_db(value, image_size, image):
    frobenius = np.linalg.norm(image)
    return ((10**(-value/10))*(frobenius**2))/image_size

def speckle_db(value, image_size, image):
    frobenius = np.linalg.norm(image)
    return ((10**(-value/10))*(frobenius**2))/image_size

def speckle_noise(value, image):
    return skimage.util.random_noise(image.copy(), mode='speckle', var = speckle_db(value, image.shape[0]*image.shape[1], image.copy()), clip=True)

def snr_percent(value):
    return 10**(-value/10)/(1+10**(-value/10))

def salt_and_pepper(value, image):
    return skimage.util.random_noise(image.copy(), mode='s&p', amount=snr_percent(value))

def salt(value, image):
    return skimage.util.random_noise(image.copy(), mode='salt', amount=snr_percent(value))

def pepper(value, image):
    return skimage.util.random_noise(image.copy(), mode='pepper', amount=snr_percent(value))

def poisson_noise(value, image):
    value = np.var(image*255)/(10**(value/10))
    noise = np.random.poisson(value * np.ones(np.shape(image)))
    return (image.copy() + np.int64(noise)/255)

def calculate_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def calculate_skewness(image):
    return skew(image.flatten())

def calculate_entropy(image):
    return shannon_entropy(image)

def calculate_energy(image):
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient
    energy = np.sqrt(sobel_h**2 + sobel_v**2)
    return (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

def calculate_kurtosis(image):
    return kurtosis(image.flatten())

def compute_correlation(img):
    correlation_matrix = np.corrcoef(img[:-1, :-1].flatten(), img[1:, 1:].flatten())
    correlation_val = correlation_matrix[0, 1]
    if np.isnan(correlation_val):
        correlation_val = 1
    return correlation_val

def tiles_img(image, ratio):
    M = image.shape[0]//ratio
    N = image.shape[1]//ratio
    tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
    return np.asarray(tiles)

def visualise_data(pathSave, Y, Z):
    y_labels = torch.argmax(Y, dim=1).numpy()
    unique_labels, label_counts = np.unique(y_labels, return_counts=True)
    plt.bar(unique_labels, label_counts)
    plt.xticks(unique_labels)  
    plt.xlabel('Type de bruit')
    plt.ylabel('Nombre d\'images')
    plt.title('Répartition des types de bruit')
    plt.savefig(pathSave + '/Repartition_bruit.png')
    plt.clf()

    data = {'Type de bruit': y_labels, 'SNR': Z.numpy()}
    df = pd.DataFrame(data)
    df = df.dropna(subset=['SNR'])
    sns.histplot(data=df, x='SNR', hue='Type de bruit', multiple='stack')
    plt.xlabel('SNR')
    plt.ylabel('Nombre d\'images')
    plt.title('Répartition des niveaux de SNR pour chaque type de bruit')
    plt.savefig(pathSave + '/Repartition_SNR.png')
    plt.clf()

def plot_loss(epochs, losses, pathSave):
    plt.title("Loss")
    plt.plot(epochs, losses, label="Loss par epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(pathSave + '/loss.png')
    plt.clf()

def print_from_metrics_dict(metrics_dict):
    noises = ["none", "gaussian", "speckle", "salt_and_pepper", "poisson"]
    total_f1score = 0
    for noise in noises:
        print(f"| {noise} | Precision: {metrics_dict[f'{noise}_precision']}, Rappel: {metrics_dict[f'{noise}_recall']}, F1 score: {metrics_dict[f'{noise}_f1_score']}")
        total_f1score += metrics_dict[f'{noise}_f1_score']
    print(f"|| TOTAL F1 : {total_f1score}, ")

def plot_metrics(metrics_history, pathSave, train_or_test):
    noises = ["none", "gaussian", "speckle", "salt_and_pepper", "poisson"]
    for noise in noises:
        precision_values = [metrics[f'{noise}_precision'] for metrics in metrics_history]
        recall_values = [metrics[f'{noise}_recall'] for metrics in metrics_history]
        f1_score_values = [metrics[f'{noise}_f1_score'] for metrics in metrics_history]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(precision_values)
        plt.title("Precision")
        plt.subplot(1, 3, 2)
        plt.plot(recall_values)
        plt.title("Recall")
        plt.subplot(1, 3, 3)
        plt.plot(f1_score_values)
        plt.title("F1 Score")
        plt.tight_layout()
        plt.savefig(pathSave + f"/{train_or_test}_{noise}_metrics.png")
        plt.clf()