import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from transformers import BertModel, BertConfig

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        """
        patience: número de épocas sin mejora antes de parar.
        min_delta: mejora mínima requerida.
        mode: 'max' para métricas como accuracy, recall, F1; 'min' para loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score <= self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode 'min'
            if score >= self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

class PositionalEncoding(nn.Module):
    """Codificación posicional para transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeAwareLogBERT(nn.Module):
    """
    LogBERT que maneja ventanas de tiempo con número variable de eventos.
    """
    def __init__(self, config, use_sequence_embeddings=False):
        super().__init__()
        self.config = config
        self.use_sequence_embeddings = config.use_sequence_embeddings

        # Para ventanas de tiempo, tenemos dos enfoques:

        # Opción A: Embedding único por ventana (promedio)
        self.classifier_simple = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )

        # Opción B: Procesar secuencia completa con transformer
        # (para cuando queremos usar la secuencia de eventos)
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        self.bert = BertModel(BertConfig(hidden_size=config.hidden_dim, num_attention_heads=config.num_heads, num_hidden_layers=config.num_layers))
        

        self.classifier_seq = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        #self.use_sequence = config.use_sequence_embeddings

    def forward(self, x, mask=None):
        """
        x: (batch_size, embedding_dim) — embedding promedio por ventana
        """
        if self.use_sequence_embeddings:
            # Procesar secuencia completa
            x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
            if mask is not None:
                src_key_padding_mask = (mask == 0)  # True donde hay padding
                x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # Invertir máscara para transformer (True donde hay padding)
            else:
                x = self.transformer(x)#, src_key_padding_mask=~mask.bool())  # (batch_size, seq_len, hidden_dim)
                
            x = x.mean(dim=1)  # Pooling promedio sobre la secuencia
            return self.classifier_seq(x)#.squeeze(-1)
        else:
            return self.classifier_simple(x)
        
        
    def train_model(self, train_loader, val_loader, pos_weight):
                # 7. Entrenamiento (CON BARRA DE PROGRESO CORREGIDA)
        print(f"\n{'='*70}")
        print("INICIANDO ENTRENAMIENTO")
        print('='*70)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stopping = EarlyStopping(patience=7, min_delta=0.001, mode='max')
        best_model_state = None
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            # Entrenamiento
            self.train()
            total_loss = 0

            # Usar tqdm con total conocido
            train_iterator = tqdm(
                train_loader,
                desc=f"Época {epoch+1} [Train]",
                unit="batch",
                total=len(train_loader)
            )

            for batch_idx, (embeddings, masks, labels, _) in enumerate(train_iterator):
                embeddings = embeddings.to(self.config.device)
                labels = labels.to(self.config.device).float()
                
                if masks is not None:
                    masks = masks.to(self.config.device)

                outputs = self.forward(embeddings, mask = masks)
                loss = criterion(outputs.squeeze(1), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Actualizar barra de progreso
                train_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })

                # Liberar memoria periódicamente
                if batch_idx % 1000 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            avg_loss = total_loss / len(train_loader)

            # Validación
            val_metrics = self.validate_model(val_loader)
            
            recall = val_metrics['recall']
            precision = val_metrics['precision']
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\nÉpoca {epoch+1} - Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.3f} | "
                  f"Val Prec: {val_metrics['precision']:.3f} | "
                  f"Val Rec: {val_metrics['recall']:.3f} | "
                  f"Val F1: {f1:.3f}")
            
            early_stopping(f1)
            if early_stopping.early_stop:
                print(f"Early stopping en epoch {epoch+1} (mejor F1: {early_stopping.best_score:.3f})")
                
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    print(f"Modelo restaurado al estado de la mejor época: {best_epoch+1}")
                break
            
            if f1 > early_stopping.best_score:
                best_model_state = self.state_dict().copy()
                best_epoch = epoch + 1
        return best_epoch, early_stopping.best_score, best_model_state

    
    def validate_model(self, val_loader):
        """Validación con manejo de errores."""
        if len(val_loader) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0}

        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for embeddings, masks, labels, _ in tqdm(val_loader, desc="Validando", leave=False):
                embeddings = embeddings.to(self.config.device)
                labels = labels.to(self.config.device)
                if masks is not None:
                    masks = masks.to(self.config.device)

                outputs = self.forward(embeddings, mask = masks)
                probs = torch.sigmoid(outputs.squeeze(1))
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if len(all_preds) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

        

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'classification_report': classification_report(all_labels, all_preds, zero_division=0)
        }
        
    def predict_model(self, test_loader):
        test_metrics = self.validate_model(test_loader)
        f1 = test_metrics['f1']
        confusion_matrix = test_metrics['confusion_matrix']
        classification_report = test_metrics['classification_report']
        print(f"\nTest Acc: {test_metrics['accuracy']:.3f} | "
              f"Test Prec: {test_metrics['precision']:.3f} | "
              f"Test Rec: {test_metrics['recall']:.3f} | "
              f"Test F1: {f1:.3f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix}")
        print(f"\nClassification Report:\n{classification_report}")
        return test_metrics
    
class LogBERT(nn.Module):
    def __init__(self, config, use_sequence_embeddings=False, task='logbert'):
        super().__init__()
        self.config = config
        self.use_sequence_embeddings = self.config.use_sequence_embeddings
        self.task = task  # 'supervised', 'reconstruction', 'logbert'

        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)


        if task in ('reconstruction', 'logbert'):
            # Decodificador para reconstruir el embedding promedio (768d)
            self.decoder = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, config.embedding_dim)
            )
            # Centro de la esfera para VHM (opcional)
            self.sphere_center = nn.Parameter(torch.randn(config.hidden_dim))

    def forward(self, x, mask=None, return_sequence=False):
        if self.use_sequence_embeddings:
            x = self.input_projection(x)            # (batch, seq, hidden)
            transformer_out = self.transformer(x, src_key_padding_mask=(mask==0) if mask is not None else None)
            # Representación de la secuencia (pooling)
            if return_sequence:
                seq_repr = transformer_out         # (batch, seq, hidden)
            else:
                seq_repr = transformer_out.mean(dim=1)  # (batch, hidden)
        else:
            # embedding promedio directo
            seq_repr = self.input_projection(x.unsqueeze(1)).squeeze(1)

        if self.task == 'reconstruction':
            # Reconstruir embedding promedio original
            reconstructed = self.decoder(seq_repr)   # (batch, 768)
            return reconstructed
        elif self.task == 'logbert':
            # Dos salidas: reconstrucción y distancia al centro
            reconstructed = self.decoder(seq_repr)
            distance = torch.norm(seq_repr - self.sphere_center, dim=1)
            return reconstructed, distance
        
    def train_model(self, train_loader, val_loader, pos_weight = None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        mse_loss = nn.MSELoss()
        best_val_loss = float('inf')
        alpha = self.config.alpha
        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta, mode=self.config.mode)
        best_state = None
        counter = 0
        for epoch in range(self.config.num_epochs):
            self.train()
            total_loss = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for embeddings, masks, _, _ in progress:
                embeddings = embeddings.to(self.config.device)
                masks = masks.to(self.config.device) if masks is not None else None
                reconstructed = self.forward(embeddings, mask=masks)   # (batch, 768)
                loss_recon = mse_loss(reconstructed, embeddings)   # compara con embedding original de la ventana
                loss = loss_recon
                if self.task == 'logbert':
                    # Si usas VHM, necesitas el distance y aplicar la pérdida de esfera
                    reconstructed, distance = self.forward(embeddings, mask=masks)
                    loss_vhm = torch.mean(distance ** 2)
                    loss = loss_recon + alpha * loss_vhm
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress.set_postfix({"Train Loss": f"{avg_loss:.4f}"})
            avg_loss = total_loss / len(train_loader)
            # Validación (pérdida)
            val_loss = self._compute_reconstruction_loss(val_loader, mse_loss)
            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping en epoch {epoch+1} (mejor F1: {early_stopping.best_score:.3f})")
                
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    print(f"Modelo restaurado al estado de la mejor época: {best_epoch+1}")
                break
            
            if val_loss < early_stopping.best_score:
                best_model_state = self.state_dict().copy()
                best_epoch = epoch + 1
        if best_state is not None:
            torch.save(best_state, 'best_unsup_model.pth')
        return best_epoch, best_val_loss, best_state
                
    def anomaly_score(self, dataloader):
        self.eval()
        scores = []
        with torch.no_grad():
            for embeddings, masks, _, _ in dataloader:
                embeddings = embeddings.to(self.config.device)
                masks = masks.to(self.config.device) if masks is not None else None
                reconstructed = self.forward(embeddings, mask=masks)
                mse = torch.mean((reconstructed - embeddings) ** 2, dim=1)
                scores.extend(mse.cpu().numpy())
        return np.array(scores)
    
    def _compute_reconstruction_loss(self, loader, criterion):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for embeddings, masks, _, _ in loader:
                embeddings = embeddings.to(self.config.device)
                masks = masks.to(self.config.device) if masks is not None else None
                reconstructed = self.forward(embeddings, mask=masks)
                loss = criterion(reconstructed, embeddings)
                total_loss += loss.item()
        return total_loss / len(loader)

    def compute_anomaly_scores(self, loader):
        """
        Retorna array de errores de reconstrucción (MSE) para todas las ventanas en loader.
        """
        self.eval()
        scores = []
        with torch.no_grad():
            for embeddings, masks, _, _ in tqdm(loader, desc="Computing anomaly scores"):
                embeddings = embeddings.to(self.config.device)
                masks = masks.to(self.config.device) if masks is not None else None
                reconstructed = self.forward(embeddings, mask=masks)
                mse = torch.mean((reconstructed - embeddings) ** 2, dim=1)  # (batch,)
                scores.extend(mse.cpu().numpy())
        return np.array(scores)