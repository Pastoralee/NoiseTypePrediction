from dataclasses import dataclass
import torch
from load_data import load_dataset
import torch.nn as nn
from networks.UNetLatent import UnetLatent
from networks.UNetBiais import UnetBiais
from networks.UNetFC import UnetFC
from networks.UNetFC3 import UnetFC3
from networks.EncoderMLP import EncoderMLP
import train as tr

@dataclass
class Args():
    pathData: str #chemin vers les données
    pathSave: str #dossier dans lequel sauvegarder les epochs
    device: torch.device 
    testSize: float #pourcentage du dataset attribué au test [0;1]
    batchSize: int #nombre de mini batchs utilisés pendant l'entraînement
    numWorkers: int #nombre de threads pour les dataloaders
    shuffle: bool #mélanger les données
    epochs: int
    visualise_data: bool #affiche des détails sur les données utilisé pour l'entraînement
    denoise_data: bool #débruite les images contenues dans les données
    include_orig: bool #inclure l'image d'origine dans les données d'entraînement
    loss: torch.nn.Module
    model_name: str #Biais, FC, Latent, MLP

args = Args(pathData = "D:\\Projet Imagerie\\data\\high-224",
            pathSave = "D:\\Projet Imagerie\\save",
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            testSize = 0.15,
            batchSize = 16,
            numWorkers = 1,
            shuffle = True,
            epochs = 30,
            visualise_data = False,
            denoise_data = True,
            include_orig = True,
            loss = nn.CrossEntropyLoss(),
            model_name="FC")

train_loader, test_loader = load_dataset(args)

match args.model_name:
    case "FC":
        if args.include_orig:
            args.model_name = "FC3"
            model = UnetFC3()
        else:
            model = UnetFC()
        model = model.to(args.device)
    case "Biais":
        model = UnetBiais()
        model = model.to(args.device)
    case "Latent":
        model = UnetLatent()
        model = model.to(args.device)
    case "MLP":
        model = EncoderMLP()
        model = model.to(args.device)
    case _:
        raise NameError(f"Ce modèle n'existe pas: {args.model_name}, choisissez parmis l'une des valeurs suivante: FC, Biais, Latent, MLP")

tr.train_unet_model(model, args, train_loader, test_loader)