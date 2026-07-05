"""
models_ruleid.py
------------------
Variantes de TimeAwareTransformer y DeepLogBaseline que reciben
secuencias de índices categóricos (rule_id) en lugar de embeddings
BERT precalculados. La representación vectorial se aprende desde
cero mediante una capa nn.Embedding entrenada junto con el resto
de la red — el mismo paradigma utilizado por DeepLog, LogAnomaly
y LogBERT sobre identificadores de template (log keys).

Esto permite comparar, bajo las mismas ventanas y el mismo
etiquetado, dos paradigmas de representación de entrada:
    1. Texto enriquecido + BERT preentrenado  (models.py)
    2. Identificador categórico + embedding aprendido (este módulo)
"""

import torch
import torch.nn as nn

from models import TrainingMixin, ModelConfig  # reuso del mixin de entrenamiento


class TimeAwareTransformerLogKey(TrainingMixin, nn.Module):
    """
    Transformer entrenado desde cero, idéntico en el cuerpo del
    encoder a TimeAwareTransformer, pero con una capa de embedding
    categórica como front-end en lugar de una proyección lineal
    de embeddings BERT precalculados.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, config: ModelConfig = None):
        super().__init__()
        self.config = config

        # PAD = índice 0 -> padding_idx evita que el gradiente
        # actualice el embedding de relleno
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """x: (batch, seq_len) índices long de rule_id."""
        x = self.embedding(x)             # (B, S, embedding_dim)
        x = self.input_projection(x)      # (B, S, hidden_dim)

        if mask is not None:
            pad_mask = (mask == 0)
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)

        x = x.mean(dim=1)
        return self.classifier(x)


class DeepLogBaselineLogKey(TrainingMixin, nn.Module):
    """
    Variante de DeepLogBaseline fiel al esquema categórico original
    de DeepLog: embedding entrenado desde cero indexado por log key
    (rule_id), seguido de un LSTM unidireccional.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 config: ModelConfig = None):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """x: (batch, seq_len) índices long de rule_id."""
        x = self.embedding(x)                  # (B, S, embedding_dim)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


class LogFormerAdapterLogKey(TrainingMixin, nn.Module):
    """
    Variante de LogFormerAdapter con embedding categórico aprendido
    desde cero como front-end, en lugar de embeddings BERT
    precalculados a partir de texto enriquecido. Replica el mismo
    Log-Attention encoder con adapters paralelos, pero recibiendo
    índices de rule_id en lugar de vectores de 768 dimensiones.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, bottleneck_dim: int = 64,
                 config: ModelConfig = None):
        super().__init__()
        self.config = config

        from models import LogAttentionEncoderLayer  # reuso del módulo base

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)

        self.layers = nn.ModuleList([
            LogAttentionEncoderLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                bottleneck_dim=bottleneck_dim, dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """x: (batch, seq_len) índices long de rule_id."""
        x = self.embedding(x)
        x = self.input_projection(x)
        pad_mask = (mask == 0) if mask is not None else None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        x = x.mean(dim=1)
        return self.classifier(x)

    def freeze_encoder_for_tuning(self):
        for layer in self.layers:
            layer.freeze_base()

    def unfreeze_encoder(self):
        for layer in self.layers:
            layer.unfreeze_base()

    def count_trainable_params(self) -> dict:
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
        embedding_params = sum(
            p.numel() for p in self.embedding.parameters() if p.requires_grad
        )
        projection_params = sum(
            p.numel() for p in self.input_projection.parameters() if p.requires_grad
        )
        return {
            "base_layers": base_params,
            "adapters":    adapter_params,
            "classifier":  classifier_params,
            "embedding":   embedding_params,
            "projection":  projection_params,
            "total":       base_params + adapter_params + classifier_params
                          + embedding_params + projection_params,
        }


MODELS_RULEID = {
    "transformer_logkey": TimeAwareTransformerLogKey,
    "deeplog_logkey":     DeepLogBaselineLogKey,
    "logformer_logkey":   LogFormerAdapterLogKey,
}


def build_model_ruleid(name: str, vocab_size: int, config: ModelConfig,
                       embedding_dim: int = 128, hidden_dim: int = 256,
                       **kwargs):
    if name not in MODELS_RULEID:
        raise ValueError(f"Modelo desconocido: '{name}'. "
                         f"Opciones: {list(MODELS_RULEID.keys())}")
    cls = MODELS_RULEID[name]
    if cls is LogFormerAdapterLogKey:
        return cls(vocab_size=vocab_size, embedding_dim=embedding_dim,
                   hidden_dim=hidden_dim, config=config, **kwargs)
    return cls(vocab_size=vocab_size, embedding_dim=embedding_dim,
              hidden_dim=hidden_dim, config=config)
