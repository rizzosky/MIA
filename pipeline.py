from preprocess import TimeBasedLogProcessor
import torch
import gc
from torch.utils.data import DataLoader
from model import TimeAwareLogBERT, LogBERT
from dataset import TimeWindowDataset, collate_time_windows
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
import numpy as np

class TimeBasedPipeline:
    def __init__(self, config):
        self.config = config
        self.processor = TimeBasedLogProcessor(config)
        
    def scan_chunks(self):
        """
        Carga los chunks desde metadata o los genera si no existen.
        """
        chunks = self.processor.load_chunk_metadata('chunk_metadata.pkl')

        if not chunks:
            print("No se encontró metadata de chunks. Generando nuevos chunks...")
            chunks = self.processor.group_files_by_time_chunks(
                [self.config.normal_logs_path],
                [self.config.incident_logs_path],
                chunk_minutes=self.config.chunk_minutes
            )
            self.processor.save_chunk_metadata(chunks)
        else:
            print(f"Metadata de chunks cargada: {len(chunks)} chunks encontrados.")

        return chunks
    
    def divide_chunks(self, chunks):
        chunk_items = list(chunks.items())

        # Separar chunks con y sin incidentes
        incident_chunks = [(k,v) for k,v in chunk_items if v['has_incidents']]
        normal_chunks   = [(k,v) for k,v in chunk_items if not v['has_incidents']]

        # Ordenar cada grupo por tiempo
        incident_chunks.sort(key=lambda x: x[1]['start_time'])
        normal_chunks.sort(key=lambda x: x[1]['start_time'])
        
        n_chunks_incident = len(incident_chunks)
        n_chunks_normal   = len(normal_chunks)
        train_end_incident = int(n_chunks_incident * 0.7)
        train_end_normal = int(n_chunks_normal * 0.7)
        val_end_incident = int(n_chunks_incident * 0.85)
        val_end_normal = int(n_chunks_normal * 0.85)

        # 80/20 de cada tipo independientemente
        #i_split = max(1, int(len(incident_chunks) * 0.8))
        #n_split = max(1, int(len(normal_chunks) * 0.8))

        train_chunks = incident_chunks[:train_end_incident] + normal_chunks[:train_end_normal]
        val_chunks   = incident_chunks[train_end_incident:val_end_incident] + normal_chunks[train_end_normal:val_end_normal]
        test_chunks  = incident_chunks[val_end_incident:] + normal_chunks[val_end_normal:]

        print(f"\nDistribución de chunks:")
        print(f"  Train: {len(train_chunks)} "
            f"({sum(1 for _,v in train_chunks if v['has_incidents'])} incidentes, "
            f"{sum(1 for _,v in train_chunks if not v['has_incidents'])} normales)")
        print(f"  Val:   {len(val_chunks)} "
            f"({sum(1 for _,v in val_chunks if v['has_incidents'])} incidentes, "
            f"{sum(1 for _,v in val_chunks if not v['has_incidents'])} normales)")
        return train_chunks, val_chunks, test_chunks
    
    def process_chunks(self,chunks, chunk_type):
        print(f"\n{'='*70}")
        print(f"PROCESANDO CHUNKS DE {chunk_type.upper()}")
        print('='*70)

        windows = []

        for chunk_key, chunk_info in chunks:
            chunk_windows = self.processor.process_time_chunk_by_timestamps(
                chunk_info, chunk_key, max_events=self.config.max_events_per_chunk
            )
            windows.extend(chunk_windows)

            normal_windows = [w for w in chunk_windows if not w['has_anomaly']]
            incident_windows = [w for w in chunk_windows if w['has_anomaly']]
            print(f"  Chunk {chunk_key}: {len(chunk_windows)} ventanas totales, {len(normal_windows)} normales, {len(incident_windows)} incidentes")

            # Liberar memoria
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nTotal ventanas {chunk_type}: {len(windows)}")
        return windows
    
    
    def build_dataloaders(self, train_chunks, val_chunks, test_chunks):
        if len(train_chunks) == 0:
            raise ValueError("No hay ventanas de entrenamiento disponibles!")
        train_windows = self.process_chunks(train_chunks, chunk_type="entrenamiento")
        val_windows = self.process_chunks(val_chunks, chunk_type="validación")
        test_windows = self.process_chunks(test_chunks, chunk_type="test")
        train_dataset = TimeWindowDataset(train_windows, use_sequence=True, max_seq_len=self.config.max_seq_len)
        val_dataset = TimeWindowDataset(val_windows, use_sequence=True, max_seq_len=self.config.max_seq_len)
        test_dataset = TimeWindowDataset(test_windows, use_sequence=True, max_seq_len=self.config.max_seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Reducir workers para evitar problemas de memoria
            #pin_memory=True,
            #collate_fn=collate_fn
            collate_fn=collate_time_windows
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            #pin_memory=True,
            #collate_fn=collate_fn
            collate_fn=collate_time_windows
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            #pin_memory=True,
            #ollate_fn=collate_fn
            collate_fn=collate_time_windows
        )

        print(f"\nDataLoaders creados:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Después de contar n_normal y n_incident
        n_normal   = sum(1 for w in train_windows if not w['has_anomaly'])
        n_incident = sum(1 for w in train_windows if w['has_anomaly'])
        pos_weight = torch.tensor([n_normal / max(n_incident, 1)]).to(self.config.device)
        return train_loader, val_loader, test_loader, pos_weight
    
    def train(self, train_loader, val_loader, pos_weight, model):
        # 7. Entrenamiento (CON BARRA DE PROGRESO CORREGIDA)
        print(f"\n{'='*70}")
        print("INICIANDO ENTRENAMIENTO")
        print('='*70)
        best_epoch, best_score, best_model_state = model.train_model(train_loader, val_loader, pos_weight)
        return best_epoch, best_score, best_model_state
    
    def test(self, test_loader, model):
        test_metrics = model.predict_model(test_loader)
        return test_metrics       

    def run(self):
        """
        Ejecuta el pipeline completo usando timestamps reales.
        VERSIÓN CORREGIDA - Maneja chunks vacíos
        """
        print("="*70)
        print("PIPELINE CON CHUNKS BASADOS EN TIMESTAMPS REALES")
        print("="*70)
        chunks = self.scan_chunks()
        train_chunks, val_chunks, test_chunks = self.divide_chunks(chunks)
        train_loader, val_loader, test_loader, pos_weight = self.build_dataloaders(train_chunks, val_chunks, test_chunks)
        model = TimeAwareLogBERT(self.config).to(self.config.device)
        best_epoch, best_f1, best_model_state = self.train(train_loader, val_loader, pos_weight, model)
        print(f"\nMejor época: {best_epoch}, Mejor F1: {best_f1:.4f}")
        test_results = self.test(test_loader, model)
        print(f"\nResultados en test: {test_results}")
        return model, test_results, best_f1
    
    def run_unsupervised(self):
        """
        Ejecuta el pipeline completo en modo no supervisado (solo normales).
        VERSIÓN CORREGIDA - Maneja chunks vacíos
        """
        print("="*70)
        print("PIPELINE NO SUPERVISADO CON CHUNKS BASADOS EN TIMESTAMPS REALES")
        print("="*70)
        chunks = self.scan_chunks()
        train_chunks, val_chunks, test_chunks = self.divide_chunks(chunks)
        
        # Filtrar solo chunks sin incidentes para entrenamiento no supervisado
        train_chunks = [(k,v) for k,v in train_chunks if not v['has_incidents']]
        val_chunks   = [(k,v) for k,v in val_chunks if not v['has_incidents']]
        test_chunks  = [(k,v) for k,v in test_chunks if not v['has_incidents']]

        train_loader, val_loader, test_loader, _ = self.build_dataloaders(train_chunks, val_chunks, test_chunks)
        model = LogBERT(self.config).to(self.config.device)
        best_epoch, best_val_loss, best_model_state = self.train(train_loader, val_loader, pos_weight=None, model=model)
        
        train_scores = model.compute_anomaly_scores(train_loader)
        threshold = np.percentile(train_scores, 95)  # Umbral basado en percentil de entrenamiento
        print(f"\nUmbral de anomalía (percentil 95 de entrenamiento): {threshold:.4f}")
        test_scores = model.compute_anomaly_scores(test_loader)
        test_labels = [window['has_anomaly'] for _, window in test_loader.dataset.windows]
        test_labels = np.array(test_labels)
        test_predictions = (test_scores > threshold).astype(int)
        
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions, zero_division=0)
        recall = recall_score(test_labels, test_predictions, zero_division=0)
        f1 = f1_score(test_labels, test_predictions, zero_division=0)
        roc_auc = roc_auc_score(test_labels, test_scores)
        
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        print(f"\nResultados en test no supervisado: {test_results}")
        return model, test_results
        