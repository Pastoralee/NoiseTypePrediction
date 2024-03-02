import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from prodigyopt import Prodigy
import metrics as mt
import util as ut

def train_unet_model(model, args, train_loader, test_loader):
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = None
    losses, epochs, metrics_dict_train_save, metrics_dict_test_save = [], [], [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for imgs, ground_truth, add_infos in tqdm(train_loader):
            img, ground_truth, add_info = imgs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float), add_infos.to(device=args.device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_img = model(img, add_info)
            loss_xy = args.loss(out_img, ground_truth)
            loss_xy.backward()
            train_loss += loss_xy.item()
            optimizer.step()
        metrics_dict_train = mt.get_metrics_dict(model, train_loader, args.device)
        print("---------- TRAIN ----------")
        ut.print_from_metrics_dict(metrics_dict_train)
        metrics_dict_test = mt.get_metrics_dict(model, test_loader, args.device)
        print("---------- TEST ----------")
        ut.print_from_metrics_dict(metrics_dict_test)
        metrics_dict_train_save.append(metrics_dict_train)
        metrics_dict_test_save.append(metrics_dict_test)
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': args.model_name}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
    ut.plot_loss(epochs, losses, args.pathSave)
    ut.plot_metrics(metrics_dict_train_save, args.pathSave, "train")
    ut.plot_metrics(metrics_dict_test_save, args.pathSave, "test")