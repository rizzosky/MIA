"""
models.py
---------
Contiene tres modelos para detección de anomalías en logs de SOC:

    1. TimeAwareTransformer  — Transformer entrenado desde cero (baseline propio)
    2. TimeAwareBERT         — BERT preentrenado (bert-base-uncased) fine-tuned
    3. DeepLogBaseline       — LSTM unidireccional (baseline del estado del arte)

Todos reciben como entrada embeddings de ventanas temporales y producen
una salida binaria (normal=0 / anómalo=1).
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from transformers import BertModel, BertConfig, AutoModel
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    embedding_dim: int   = 128
    hidden_dim: int      = 256
    num_heads: int       = 4
    num_layers: int      = 2
    num_epochs: int      = 30
    learning_rate: float = 2e-5
    use_sequence_embeddings: bool = True
    device: str          = "cpu"
    bert_model_name: str = "bert-base-uncased"


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode="max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return
        improved = (score > self.best_score + self.min_delta
                    if self.mode == "max"
                    else score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ─────────────────────────────────────────────────────────────────────────────
# Mixin de entrenamiento compartido
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMixin:
    """Lógica de entrenamiento y validación compartida por todos los modelos."""

    def _compute_metrics(self, all_labels, all_preds, total_loss, num_batches):
        val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if len(all_preds) == 0:
            return {k: 0 for k in
                    ["accuracy", "precision", "recall", "f1",
                     "roc_auc", "val_loss"]}
        return {
            "accuracy":              accuracy_score(all_labels, all_preds),
            "precision":             precision_score(all_labels, all_preds,
                                                     zero_division=0),
            "recall":                recall_score(all_labels, all_preds,
                                                  zero_division=0),
            "f1":                    f1_score(all_labels, all_preds,
                                              zero_division=0),
            "roc_auc":               roc_auc_score(all_labels, all_preds),
            "confusion_matrix":      confusion_matrix(all_labels, all_preds),
            "classification_report": classification_report(all_labels, all_preds,
                                                           zero_division=0),
            "val_loss":              val_loss,
        }

    def validate_model(self, val_loader, criterion=None):
        self.eval()
        all_preds, all_labels = [], []
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando", leave=False):
                embeddings, masks, labels, _ = batch
                embeddings = embeddings.to(self.config.device)
                labels     = labels.to(self.config.device).float()
                masks      = masks.to(self.config.device) if masks is not None else None

                outputs = self.forward(embeddings, mask=masks)
                probs   = torch.sigmoid(outputs.squeeze(1))
                preds   = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if criterion is not None:
                    loss = criterion(outputs.squeeze(1), labels)
                    total_loss += loss.item()
                    num_batches += 1

        return self._compute_metrics(all_labels, all_preds,
                                     total_loss, num_batches)

    def predict_model(self, test_loader):
        import time
        infer_start = time.time()
        metrics = self.validate_model(test_loader)
        infer_time = time.time() - infer_start

        n_samples = sum(len(batch[2]) for batch in test_loader)
        ms_per_sample = (infer_time / n_samples) * 1000 if n_samples else 0
        self.inference_time_total = infer_time
        self.inference_ms_per_sample = ms_per_sample

        print(f"\nTest Acc:  {metrics['accuracy']:.3f} | "
              f"Prec: {metrics['precision']:.3f} | "
              f"Rec:  {metrics['recall']:.3f} | "
              f"F1:   {metrics['f1']:.3f}")
        print(f"Tiempo de inferencia: {infer_time:.3f}s total, "
              f"{ms_per_sample:.3f} ms/ventana ({n_samples} ventanas)")
        print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")
        return metrics

    def train_model(self, train_loader, val_loader, pos_weight):
        import time
        print(f"\n{'='*70}")
        print(f"ENTRENANDO: {self.__class__.__name__}")
        print("=" * 70)

        self.train_losses, self.val_losses = [], []
        self.val_accs, self.val_precs      = [], []
        self.val_recs, self.val_f1s        = [], []
        self.epoch_times                   = []

        optimizer      = torch.optim.AdamW(self.parameters(),
                                           lr=self.config.learning_rate)
        criterion      = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        early_stopping = EarlyStopping(patience=7, min_delta=0.001, mode="max")
        best_model_state, best_epoch = None, 0
        training_start = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.train()
            total_loss = 0.0

            pbar = tqdm(train_loader,
                        desc=f"Época {epoch+1} [Train]",
                        unit="batch", total=len(train_loader))

            for batch_idx, (embeddings, masks, labels, _) in enumerate(pbar):
                embeddings = embeddings.to(self.config.device)
                labels     = labels.to(self.config.device).float()
                masks      = masks.to(self.config.device) if masks is not None else None

                outputs = self.forward(embeddings, mask=masks)
                loss    = criterion(outputs.squeeze(1), labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({
                    "loss":     f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                })

                if batch_idx % 1000 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)

            val_metrics = self.validate_model(val_loader, criterion=criterion)
            prec = val_metrics["precision"]
            rec  = val_metrics["recall"]
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            self.val_losses.append(val_metrics["val_loss"])
            self.val_accs.append(val_metrics["accuracy"])
            self.val_precs.append(prec)
            self.val_recs.append(rec)
            self.val_f1s.append(f1)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            print(f"\nÉpoca {epoch+1} — Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.3f} | "
                  f"Val Prec: {prec:.3f} | "
                  f"Val Rec: {rec:.3f} | "
                  f"Val F1: {f1:.3f} | "
                  f"Tiempo: {epoch_time:.2f}s")

            early_stopping(f1)
            if f1 > (early_stopping.best_score or 0):
                best_model_state = {k: v.clone()
                                    for k, v in self.state_dict().items()}
                best_epoch = epoch + 1

            if early_stopping.early_stop:
                print(f"Early stopping en época {epoch+1} "
                      f"(mejor F1: {early_stopping.best_score:.3f})")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    print(f"Modelo restaurado a la época {best_epoch}")
                break

        total_time = time.time() - training_start
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f"\nTiempo total de entrenamiento: {total_time:.2f}s "
              f"({len(self.epoch_times)} épocas, "
              f"promedio {avg_epoch_time:.2f}s/época)")
        self.total_training_time = total_time
        self.avg_epoch_time      = avg_epoch_time

        return best_epoch, early_stopping.best_score, best_model_state


# ─────────────────────────────────────────────────────────────────────────────
# 1. TimeAwareTransformer — transformer desde cero (tu modelo actual)
# ─────────────────────────────────────────────────────────────────────────────

class TimeAwareTransformer(TrainingMixin, nn.Module):
    """
    Transformer entrenado desde cero sobre embeddings de ventanas temporales.
    Sirve como baseline propio y permite comparar el valor del preentrenamiento
    de BERT sobre los mismos datos.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=config.num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, embedding_dim)  si use_sequence_embeddings=True
               (batch, embedding_dim)           si use_sequence_embeddings=False
        mask : (batch, seq_len) — 1 donde hay evento, 0 donde hay padding
        """
        if x.dim() == 2:
            # Modo simple: embedding promedio ya calculado
            return self.classifier(x)

        x = self.input_projection(x)          # (B, S, H)
        if mask is not None:
            pad_mask = (mask == 0)            # True donde hay padding
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)
        x = x.mean(dim=1)                    # pooling promedio
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TimeAwareBERT — BERT preentrenado + clasificador
# ─────────────────────────────────────────────────────────────────────────────

class TimeAwareBERT(TrainingMixin, nn.Module):
    """
    Usa bert-base-uncased como encoder de la secuencia de embeddings de eventos.

    El pipeline de entrada es idéntico al de TimeAwareTransformer:
    cada evento ya está representado como un vector de dimensión embedding_dim.
    BERT opera como encoder contextual bidireccional sobre la secuencia de
    eventos de la ventana temporal, reemplazando al TransformerEncoder propio.

    Se usa el token [CLS] como representación de la ventana completa,
    consistente con el uso estándar de BERT para clasificación de secuencias.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Proyección de embedding_dim → bert hidden_size (768 para bert-base)
        bert_hidden = 768
        self.input_projection = nn.Linear(config.embedding_dim, bert_hidden)

        # BERT preentrenado — se fine-tunea junto con el clasificador
        self.bert = AutoModel.from_pretrained(config.bert_model_name)

        # Clasificador sobre el token [CLS]
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bert_hidden // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, embedding_dim)
        mask : (batch, seq_len) — attention mask: 1=evento real, 0=padding
        """
        # Proyectar al espacio de BERT
        x = self.input_projection(x)          # (B, S, 768)

        # BERT espera inputs_embeds en lugar de input_ids
        # cuando se le pasan embeddings ya calculados
        attention_mask = mask if mask is not None else torch.ones(
            x.shape[:2], dtype=torch.long, device=x.device
        )

        outputs = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )

        # Usar el token [CLS] (posición 0) como representación de la ventana
        cls_output = outputs.last_hidden_state[:, 0, :]   # (B, 768)
        return self.classifier(cls_output)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DeepLogBaseline — LSTM unidireccional (baseline estado del arte)
# ─────────────────────────────────────────────────────────────────────────────

class DeepLogBaseline(TrainingMixin, nn.Module):
    """
    Adaptación de DeepLog para clasificación binaria supervisada sobre
    ventanas temporales. El modelo original usa LSTM para predecir el
    siguiente evento; aquí se adapta para clasificar la ventana completa
    como normal o anómala, usando los mismos embeddings de entrada que
    TimeAwareTransformer y TimeAwareBERT para permitir comparación directa.

    Diferencias respecto al DeepLog original:
        - Entrada: embeddings de eventos (no one-hot de log keys)
        - Salida: clasificación binaria (no predicción del siguiente evento)
        - Entrenamiento: supervisado con etiquetas (no semi-supervisado)
    Estas diferencias se deben a la naturaleza del dataset disponible
    (datos etiquetados por analistas) y se documentan en la sección
    de comparación de modelos.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.1 if config.num_layers > 1 else 0.0,
            bidirectional=False,   # DeepLog es unidireccional
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, embedding_dim)
        mask : no se usa en LSTM — incluido por compatibilidad de interfaz
        """
        lstm_out, (h_n, _) = self.lstm(x)
        # Usar el último hidden state como representación de la secuencia
        last_hidden = h_n[-1]                 # (B, hidden_dim)
        return self.classifier(last_hidden)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LogFormerAdapter — Log-Attention encoder + parallel adapters
# ─────────────────────────────────────────────────────────────────────────────

class ParallelAdapter(nn.Module):
    """
    Adapter paralelo (bottleneck down-projection -> activación -> up-projection)
    insertado junto a cada capa del encoder, siguiendo el diseño de
    LogFormer (Guo et al., 2024). A diferencia de un adapter serial,
    el adapter paralelo se suma a la salida de la capa en lugar de
    insertarse en el flujo secuencial, lo que permite congelar el
    encoder base y entrenar únicamente los adapters durante el tuning.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class LogAttentionEncoderLayer(nn.Module):
    """
    Una capa del Log-Attention encoder de LogFormer: una capa estándar
    de TransformerEncoder con un adapter paralelo conectado a su salida.
    Durante el pre-entrenamiento, tanto la capa base como el adapter
    son entrenables. Durante el tuning, la capa base se congela
    (requires_grad=False) y solo el adapter se actualiza.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.base_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.adapter = ParallelAdapter(hidden_dim, bottleneck_dim)

    def forward(self, x, src_key_padding_mask=None):
        base_out    = self.base_layer(x, src_key_padding_mask=src_key_padding_mask)
        adapter_out = self.adapter(x)
        return base_out + adapter_out

    def freeze_base(self):
        """Congela la capa base, dejando entrenable únicamente el adapter."""
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base_layer.parameters():
            param.requires_grad = True


class LogFormerAdapter(TrainingMixin, nn.Module):
    """
    Adaptación de LogFormer (Guo et al., 2024) al escenario de este
    trabajo. LogFormer original propone dos etapas: (1) pre-entrenamiento
    de un Log-Attention encoder sobre un dominio fuente para capturar
    semántica compartida entre dominios de logs, y (2) tuning en el
    dominio objetivo donde el encoder se congela y solo se entrenan
    adapters paralelos insertados en cada capa, reduciendo
    drásticamente los parámetros entrenables en la etapa de adaptación.

    En este trabajo, el dominio fuente se define como el comportamiento
    normal general del sistema, y el dominio objetivo como la tarea de
    discriminar ventanas normales de ventanas anómalas del incidente
    específico. La etapa de pre-entrenamiento utiliza el mismo dataset
    de entrenamiento que los otros modelos; la etapa de tuning congela
    el Log-Attention encoder resultante y reentrena únicamente los
    adapters y el clasificador final, replicando el espíritu de
    transferencia de conocimiento con bajo costo de parámetros del
    trabajo original.

    Diferencias respecto al LogFormer original:
        - Encoder de extracción de features: se reutiliza el mismo
          embedding BERT (bert-base-uncased) usado en los demás
          modelos, en lugar de Sentence-BERT, para mantener la
          comparación bajo las mismas condiciones de entrada.
        - Log-Attention module: se omite la incorporación de
          parámetros extraídos del log parsing (P_i), ya que el
          dataset de este trabajo no cuenta con un parser de
          parámetros estructurado equivalente al usado en el
          paper original.
        - Dominio fuente/objetivo: en el paper original corresponden
          a datasets distintos (p. ej. HDFS -> BGL); en este trabajo
          corresponden a comportamiento normal general vs. el
          incidente específico, dado que solo se dispone de un
          dominio real.
    """

    def __init__(self, config: ModelConfig, num_layers: int = 2,
                 bottleneck_dim: int = 64):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)

        self.layers = nn.ModuleList([
            LogAttentionEncoderLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                bottleneck_dim=bottleneck_dim,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        # Clasificador de una sola capa lineal, consistente con el
        # diseño original ("The classifier is simply implemented by
        # one linear layer").
        self.classifier = nn.Linear(config.hidden_dim, 1)

        self._tuning_mode = False

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        pad_mask = (mask == 0) if mask is not None else None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        x = x.mean(dim=1)  # pooling promedio sobre la secuencia
        return self.classifier(x)

    def freeze_encoder_for_tuning(self):
        """
        Inicia la etapa de adapter-based tuning: congela las capas
        base del Log-Attention encoder, dejando entrenables
        únicamente los adapters y el clasificador final.
        """
        for layer in self.layers:
            layer.freeze_base()
        self._tuning_mode = True

    def unfreeze_encoder(self):
        for layer in self.layers:
            layer.unfreeze_base()
        self._tuning_mode = False

    def count_trainable_params(self) -> dict:
        """Desglosa los parámetros entrenables por componente."""
        adapter_params = sum(
            p.numel() for layer in self.layers
            for p in layer.adapter.parameters() if p.requires_grad
        )
        base_params = sum(
            p.numel() for layer in self.layers
            for p in layer.base_layer.parameters() if p.requires_grad
        )
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )
        projection_params = sum(
            p.numel() for p in self.input_projection.parameters() if p.requires_grad
        )
        return {
            "base_layers": base_params,
            "adapters":    adapter_params,
            "classifier":  classifier_params,
            "projection":  projection_params,
            "total":       base_params + adapter_params
                          + classifier_params + projection_params,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory — instanciar cualquier modelo por nombre
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "transformer": TimeAwareTransformer,
    "bert":        TimeAwareBERT,
    "deeplog":     DeepLogBaseline,
    "logformer":   LogFormerAdapter,
}

def build_model(name: str, config: ModelConfig) -> nn.Module:
    """
    Uso:
        config = ModelConfig(device="mps")
        model  = build_model("bert", config).to(config.device)
    """
    if name not in MODELS:
        raise ValueError(f"Modelo desconocido: '{name}'. "
                         f"Opciones: {list(MODELS.keys())}")
    return MODELS[name](config)