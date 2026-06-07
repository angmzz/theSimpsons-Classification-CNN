import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_all_preds(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
    return np.array(all_preds), np.array(all_targets)

def plot_training_metrics(history, title='Métricas de Entrenamiento'):
    if history is None or history.empty:
        print("Historial vacío. No hay métricas para graficar.")
        return
        
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    if 'train/loss' in df.columns:
        sns.lineplot(data=df, x='epoch', y='train/loss', ax=ax[0], label='Train Loss', marker='o', linewidth=2)
    if 'val/loss' in df.columns:
        sns.lineplot(data=df, x='epoch', y='val/loss', ax=ax[0], label='Val Loss', marker='o', linewidth=2)
    ax[0].set_title(f'{title} - Evolución del Loss', fontsize=14, pad=10)
    ax[0].set_xlabel('Epoch', fontsize=12)
    ax[0].set_ylabel('Loss', fontsize=12)
    
    # Accuracy
    if 'train/acc' in df.columns:
        sns.lineplot(data=df, x='epoch', y='train/acc', ax=ax[1], label='Train Acc', marker='o', linewidth=2)
    if 'val/acc' in df.columns:
        sns.lineplot(data=df, x='epoch', y='val/acc', ax=ax[1], label='Val Acc', marker='o', linewidth=2)
    ax[1].set_title(f'{title} - Evolución del Accuracy', fontsize=14, pad=10)
    ax[1].set_xlabel('Epoch', fontsize=12)
    ax[1].set_ylabel('Accuracy', fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(targets, preds, active_classes, class_names, title="Matriz de Confusión"):
    cm = confusion_matrix(targets, preds, labels=active_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 8},
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title(f"{title} - Matriz de Confusión", fontsize=16)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_gradcam_samples(model, dataloader, preds, targets, target_layer, idx_to_class, num_samples=3):
    correct_indices = np.where(preds == targets)[0]
    incorrect_indices = np.where(preds != targets)[0]
    
    # Seleccionar aleatoriamente muestras
    np.random.seed(42)
    num_correct = min(num_samples, len(correct_indices))
    num_incorrect = min(num_samples, len(incorrect_indices))
    
    selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if len(correct_indices) > 0 else []
    selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if len(incorrect_indices) > 0 else []
    
    selected_indices = np.concatenate([selected_correct, selected_incorrect])
    if len(selected_indices) == 0:
        return
        
    # Extraer el modelo interno puro (sin el wrapper de Lightning)
    inner_model = model.model if hasattr(model, 'model') else model
    inner_model.eval()
    
    # Descongelar temporalmente para que Grad-CAM pueda calcular gradientes espaciales
    for param in inner_model.parameters():
        param.requires_grad = True
        
    cam = GradCAM(model=inner_model, target_layers=[target_layer])
    
    fig, axes = plt.subplots(2, max(len(selected_indices), 1), figsize=(4 * len(selected_indices), 8))
    if len(selected_indices) == 1:
        axes = np.array([axes]).T 
        
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    count = 0
    for idx in selected_indices:
        img_tensor, true_label = dataloader.dataset[idx]
        pred_label = preds[idx]

        img_input = img_tensor.unsqueeze(0).to(model.device)
        img_input.requires_grad_(True)
        
        grayscale_cam = cam(input_tensor=img_input, targets=None)[0, :]
        
        img_np = img_tensor.cpu().numpy().transpose((1, 2, 0))
        img_rgb = std * img_np + mean
        img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
        
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        
        title_text = f"Pred: {idx_to_class[pred_label]}\nReal: {idx_to_class[true_label]}"
        color = 'green' if pred_label == true_label else 'red'
        
        # Fila 1: Imagen Original
        axes[0, count].imshow(img_rgb)
        axes[0, count].set_title(title_text, color=color, fontsize=10)
        axes[0, count].axis('off')
        
        # Fila 2: Grad-CAM
        axes[1, count].imshow(visualization)
        axes[1, count].axis('off')
        
        count += 1
        
    plt.suptitle("Análisis Grad-CAM: Aciertos (Verde) y Errores (Rojo)", fontsize=16)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, trainer, datamodule, history, target_layer, title="CNN"):
    """
    Función orquestadora que centraliza toda la evaluación visual.
    1. Llama a trainer.test()
    2. Imprime Curvas de Entrenamiento
    3. Extrae todas las predicciones del test set
    4. Imprime Matriz de Confusión
    5. Imprime grilla de Grad-CAM
    """
    print(f"\n{'='*50}\nEvaluando {title}\n{'='*50}")
    
    # 1. Test
    trainer.test(model, datamodule=datamodule)
    
    # 2. Curvas
    plot_training_metrics(history, title=title)
    
    # 3. Extraer preds
    test_loader = datamodule.test_dataloader()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    preds, targets = get_all_preds(model, test_loader)
    
    # Info de clases
    active_classes = sorted(list(set(targets)))
    idx_to_class = {i: k for i, k in enumerate(datamodule.classes)}
    class_names = [idx_to_class[i] for i in active_classes]
    
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(targets, preds, labels=active_classes, target_names=class_names, zero_division=0))
    
    # 4. Matriz
    plot_confusion_matrix(targets, preds, active_classes, class_names, title=title)
    
    # 5. Grad-CAM
    if target_layer is not None:
        plot_gradcam_samples(model, test_loader, preds, targets, target_layer, idx_to_class, num_samples=3)
