from tkinter import Tk
from tkinter import filedialog
import torch
from networks.UNetLatent import UnetLatent
from networks.UNetBiais import UnetBiais
from networks.UNetFC import UnetFC
from networks.UNetFC3 import UnetFC3
from networks.EncoderMLP import EncoderMLP
import load_data as ld
import csv
import os
import cv2
import numpy as np
import util as ut
import metrics as mt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def ask_data_and_save_path():
    print("Veuillez choisir le dossier contenant vos données:")
    pathData = filedialog.askdirectory()
    print("Dossier données choisi: ", pathData)
    print("Veuillez choisir le dossier dans lequel sauvegarder vos resultats:")
    pathSave = filedialog.askdirectory() + "/"
    print("Dossier sauvegarde choisi: ", pathSave)
    return pathData, pathSave

def save_result_to_csv(model, modelName, pathData, pathSave, device):
    with open(pathSave + '/resultats_predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Nom de l\'image', 'Pas de bruit', 'Gaussien', 'Speckle', 'Salt and Pepper', 'Poisson'])
        for fichier in os.listdir(pathData):
            chemin_fichier = os.path.join(pathData, fichier)
            img = cv2.imread(chemin_fichier, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            addInfo = ld.get_additional_image_info(img)
            _, _, _, cD = ut.calculate_dwt(img)
            img_energy = ut.calculate_energy(img)
            img_energy, cD = torch.from_numpy(np.expand_dims(cv2.resize(img_energy, (224, 224)), 0)), torch.from_numpy(np.expand_dims(cv2.resize(cD, (224, 224)), 0))
            if modelName == 'FC3':
                img_network = torch.cat((torch.from_numpy(np.expand_dims(cv2.resize(img, (224, 224)), 0)), img_energy, cD), dim=0)
            else:
                img_network = torch.cat((img_energy, cD), dim=0)
            img_network, addInfo = torch.unsqueeze(img_network, 0).to(device=device, dtype=torch.float), torch.unsqueeze(addInfo, 0).to(device=device, dtype=torch.float)
            noise_pred = model(img_network, addInfo)
            writer.writerow([fichier, *noise_pred.cpu().detach().numpy().tolist()])

def compute_confusion_matrix(model, modelName, pathData, device, SNR, noise_auto, denoise_input):
    files = os.listdir(pathData)
    preds, labels = torch.empty(len(files)), torch.empty(len(files))
    for i, fichier in enumerate(tqdm(files)):
        chemin_fichier = os.path.join(pathData, fichier)
        img = cv2.imread(chemin_fichier, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        if denoise_input:
            img = ld.NLM_denoising(img)
        if noise_auto:
            img, label = ld.apply_random_noise(img, SNR)
            labels[i] = torch.argmax(label).item()
        else:
            labels[i] = 0
        addInfo = ld.get_additional_image_info(img)
        _, _, _, cD = ut.calculate_dwt(img)
        img_energy = ut.calculate_energy(img)
        img_energy, cD = torch.from_numpy(np.expand_dims(cv2.resize(img_energy, (224, 224)), 0)), torch.from_numpy(np.expand_dims(cv2.resize(cD, (224, 224)), 0))
        if modelName == 'FC3':
            img_network = torch.cat((torch.from_numpy(np.expand_dims(cv2.resize(img, (224, 224)), 0)), img_energy, cD), dim=0)
        else:
            img_network = torch.cat((img_energy, cD), dim=0)
        img_network, addInfo = torch.unsqueeze(img_network, 0).to(device=device, dtype=torch.float), torch.unsqueeze(addInfo, 0).to(device=device, dtype=torch.float)
        noise_pred = model(img_network, addInfo)
        noise_pred = mt.normalize_max_value(noise_pred)
        preds[i] = torch.argmax(noise_pred).item()
    print(confusion_matrix(labels, preds))
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tk().withdraw()
print("Veuillez choisir un modele pour effectuer la prédiction du bruit:")
filename = filedialog.askopenfilename()
print("Modele choisi: ", filename)
pathData, pathSave = ask_data_and_save_path()
checkpoint = torch.load(filename, map_location=device)
match checkpoint['model_name']:
    case 'FC':
        model = UnetFC()
    case 'Biais':
        model = UnetBiais()
    case 'Latent':
        model = UnetLatent()
    case 'MLP':
        model = EncoderMLP()
    case 'FC3':
        model = UnetFC3()
    case _:
        raise Exception(f"Vous n'avez pas choisi un modèle correct : {checkpoint['model_name']} n'est pas reconnu") 
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

#save_result_to_csv(model, checkpoint['model_name'], pathData, pathSave, device)
compute_confusion_matrix(model, checkpoint['model_name'], pathData, device, [25], noise_auto=True, denoise_input=True)