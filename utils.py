import matplotlib.pyplot as plt

import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(model, save_path='training_curves.png'):
    epochs = range(1, len(model.train_losses) + 1)

    plt.figure(figsize=(12, 5))
    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, model.train_losses, 'b-', label='Train Loss')
    if hasattr(model, 'val_losses') and model.val_losses:
        plt.plot(epochs, model.val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss curves')

    # Métricas
    plt.subplot(1, 2, 2)
    plt.plot(epochs, model.val_accs, 'g-', label='Accuracy')
    plt.plot(epochs, model.val_precs, 'c-', label='Precision')
    plt.plot(epochs, model.val_recs, 'm-', label='Recall')
    plt.plot(epochs, model.val_f1s, 'y-', label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation metrics')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    



def plot_confusion_matrix(model, test_loader, threshold=0.5, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, masks, labels, _ in test_loader:
            embeddings = embeddings.to(model.config.device)
            masks = masks.to(model.config.device) if masks is not None else None
            labels = labels.to(model.config.device)
            outputs = model.forward(embeddings, mask=masks)
            probs = torch.sigmoid(outputs.squeeze(1))
            preds = (probs > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Incident'])
    disp.plot(cmap='Blues')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz de confusión guardada en {save_path}")
    else:
        plt.show()
    plt.close()

def analyze_errors(test_loader, model, threshold=0.5, top_n=5):
    model.eval()
    errors = {'fp': [], 'fn': []}   # cada elemento: (prob, meta)
    with torch.no_grad():
        for embeddings, masks, labels, metas in test_loader:
            embeddings = embeddings.to(model.config.device)
            masks = masks.to(model.config.device) if masks is not None else None
            labels = labels.to(model.config.device)
            outputs = model.forward(embeddings, mask=masks)
            probs = torch.sigmoid(outputs.squeeze(1)).cpu().numpy()
            preds = (probs > threshold).astype(int)
            for i in range(len(labels)):
                true = labels[i].item()
                pred = preds[i]
                if true == 0 and pred == 1:
                    errors['fp'].append((probs[i], metas[i]))
                elif true == 1 and pred == 0:
                    errors['fn'].append((probs[i], metas[i]))
    # Ordenar FP por probabilidad descendente (los más confiados en ser incidente pero son normales)
    errors['fp'].sort(key=lambda x: x[0], reverse=True)
    # Ordenar FN por probabilidad ascendente (los que el modelo tuvo menos confianza en que son incidente)
    errors['fn'].sort(key=lambda x: x[0])
    
    print(f"Top-{top_n} False Positives (mayor confianza de incidente, pero era normal):")
    for i, (prob, meta) in enumerate(errors['fp'][:top_n]):
        print(f"  {i+1}: score={prob:.3f}, window_time={meta.get('start_time', '?')} - {meta.get('end_time', '?')}, "
              f"n_events={meta.get('n_events', 0)}")
        # Opcional: mostrar técnicas MITRE si las hay (aunque no debería haber)
    
    print(f"\nTop-{top_n} False Negatives (menor confianza de incidente, pero era incidente):")
    for i, (prob, meta) in enumerate(errors['fn'][:top_n]):
        mitre = meta.get('mitre_techniques', [])
        print(f"  {i+1}: score={prob:.3f}, MITRE={mitre if mitre else 'N/A'}, "
              f"time={meta.get('start_time', '?')} - {meta.get('end_time', '?')}")
    return errors
