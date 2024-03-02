import torch

def get_noise_type(index):
    match index:
        case 0:
            return "none"
        case 1:
            return "gaussian"
        case 2:
            return "speckle"
        case 3:
            return "salt_and_pepper"
        case 4:
            return "poisson"
        case _:
            raise ValueError("Erreur indexage bruit")

def generate_dict():
    return {"none_TP": 0, "none_FP": 0, "none_FN": 0,
            "gaussian_TP": 0, "gaussian_FP": 0, "gaussian_FN": 0,
            "speckle_TP": 0, "speckle_FP": 0, "speckle_FN": 0,
            "salt_and_pepper_TP": 0, "salt_and_pepper_FP": 0, "salt_and_pepper_FN": 0,
            "poisson_TP": 0, "poisson_FP": 0, "poisson_FN": 0}

def normalize_max_value(tensor):
    max_values, _ = torch.max(tensor, dim=-1, keepdim=True)
    normalized_tensor = torch.zeros_like(tensor)
    normalized_tensor[tensor == max_values] = 1
    return normalized_tensor

def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_metrics(metrics_dict):
    result_metrics = {}
    noises = ["none", "gaussian", "speckle", "salt_and_pepper", "poisson"]
    for noise in noises:
        tp_key = f"{noise}_TP"
        fp_key = f"{noise}_FP"
        fn_key = f"{noise}_FN"
        precision = calculate_precision(metrics_dict[tp_key], metrics_dict[fp_key])
        recall = calculate_recall(metrics_dict[tp_key], metrics_dict[fn_key])
        f1_score = calculate_f1_score(precision, recall)
        result_metrics[f"{noise}_precision"] = precision
        result_metrics[f"{noise}_recall"] = recall
        result_metrics[f"{noise}_f1_score"] = f1_score
    return result_metrics

def get_metrics_dict(model, dataloader, device):
    model.eval()
    metrics_dict = generate_dict()
    for imgs, ground_truth, add_infos in dataloader.dataset:
        img, ground_truth, add_info = imgs.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float), add_infos.to(device=device, dtype=torch.float)
        out_vector = model(torch.unsqueeze(img, 0), torch.unsqueeze(add_info, 0))
        out_vector = torch.squeeze(normalize_max_value(out_vector))
        for i in range(5):
            if ground_truth[i] == 0 and out_vector[i] == 0:
                continue
            if ground_truth[i] == 1 and out_vector[i] == 1:
                metrics_dict[f"{get_noise_type(i)}_TP"] += 1
            elif ground_truth[i] == 0 and out_vector[i] == 1:
                metrics_dict[f"{get_noise_type(i)}_FP"] += 1
            elif ground_truth[i] == 1 and out_vector[i] == 0:
                metrics_dict[f"{get_noise_type(i)}_FN"] += 1
    return calculate_metrics(metrics_dict)