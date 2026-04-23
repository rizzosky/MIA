from pipeline import TimeBasedPipeline
from config import RealTimeConfig
import torch
import os
from datetime import datetime

if __name__ == "__main__":
    config = RealTimeConfig()
    pipeline = TimeBasedPipeline(config)
    #os.remove('chunk_metadata.pkl') if os.path.exists('chunk_metadata.pkl') else None

    # Ejecutar pipeline con timestamps reales
    #model, test_results, best_f1 = pipeline.run()
    model, test_results, best_f1 = pipeline.run()
    # Guardar modelo y resultados
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_f1': best_f1
        #'mitre_results': dict(mitre_results)
    }, os.path.join(config.model_save_path, f"lbert_supervised_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"))

    print(f"\n Modelo guardado en {config.model_save_path}bert_supervised_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")