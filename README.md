# Scripts del Proyecto — TimeAwareBERT

Inventario completo de los scripts utilizados para el preprocesamiento,
entrenamiento y análisis de resultados de la tesis. Organizados por
etapa del pipeline.

---

## 1. Preprocesamiento de datos

### `prepare_dataset.py`
Etapa 1 del pipeline principal (representación rica: texto enriquecido
+ BERT preentrenado). Lee los logs JSONL ya filtrados por `jq`, genera
embeddings de 768 dimensiones con `bert-base-uncased`, construye
ventanas temporales deslizantes y aplica el split estratificado por
clase. Soporta submuestreo uniforme en el tiempo vía
`--max_events_per_class` para sistemas de alto volumen (ver
`extract_pfsense.sh`). Incluye instrumentación de tiempo por etapa.

```bash
python prepare_dataset.py \
    --normal_path   <carpeta_normal> \
    --incident_path <carpeta_incidente> \
    --output        ./data/windows.pkl \
    --window_minutes 5 --step_minutes 1 \
    --device mps [--max_events_per_class N]
```

### `prepare_dataset_ruleid.py`
Variante de `prepare_dataset.py` para el experimento de representación
categórica (esquema clásico DeepLog/LogAnomaly/LogBERT): cada evento
se representa por su `rule_id` mapeado a un índice de vocabulario
(embedding aprendido desde cero, sin BERT). Construye el vocabulario
únicamente a partir del split de entrenamiento para evitar leakage.

```bash
python prepare_dataset_ruleid.py \
    --normal_path   <carpeta_normal> \
    --incident_path <carpeta_incidente> \
    --output        ./data/windows_ruleid.pkl
```

### `extract_task_scheduler.sh`
Script de extracción `jq` + `parallel` para el experimento
principal: filtra el agente `256` (Windows Task Scheduler,
`location = EventChannel`) desde los archivos crudos de Wazuh,
para ambos períodos, extrayendo los campos documentados en la
Metodología (incluidos `process_id` y `thread_id`, requeridos por
la construcción del texto de entrada). Ajustar las rutas al inicio
del script antes de ejecutar.

```bash
chmod +x extract_task_scheduler.sh
./extract_task_scheduler.sh
```

### `extract_pfsense.sh`
Script de extracción `jq` + `parallel` para el experimento de
generalización a pfSense: filtra el agente `016` combinando sus dos
`location` (`rsyslog` y `pfBlocker`) desde los archivos crudos de
Wazuh, tanto para el período normal como el de incidente. Ajustar las
rutas al inicio del script antes de ejecutar.

```bash
chmod +x extract_pfsense.sh
./extract_pfsense.sh
```

### `check_seqlen.py`
Script de diagnóstico: calcula la distribución (media, mediana,
percentiles) de eventos por ventana en un dataset `.pkl` ya generado,
sin necesidad de recalcular embeddings. Se usó para diagnosticar el
problema de truncamiento de secuencia en el experimento de pfSense
(Sección 5.x de la tesis).

```bash
python check_seqlen.py <path_al_pkl>
```

---

## 2. Datasets y modelos (representación rica: texto + BERT)

### `dataset.py`
Define `TimeWindowDataset` y `collate_time_windows`. **Contiene la
corrección de muestreo uniforme** (en lugar de truncamiento a los
primeros N eventos) aplicada durante el diagnóstico del experimento
de pfSense — ver Sección de Metodología de la tesis. Soporta modo
"embedding promedio" y modo "secuencia completa".

### `models.py`
Define las arquitecturas de los cuatro modelos con representación
rica:
- `TimeAwareTransformer` — transformer entrenado desde cero
- `TimeAwareBERT` — BERT preentrenado con fine-tuning completo
- `DeepLogBaseline` — LSTM unidireccional (adaptación supervisada de DeepLog)
- `LogFormerAdapter` — Log-Attention encoder con adapters paralelos
  (pre-entrenamiento + adapter tuning, adaptación de LogFormer)

También define `TrainingMixin` (lógica común de entrenamiento,
validación y medición de tiempos) y `ModelConfig`.

---

## 3. Datasets y modelos (representación categórica: rule_id)

### `dataset_ruleid.py`
Define `RuleIdWindowDataset` y `collate_ruleid_windows`, análogos a
los de `dataset.py` pero para secuencias de índices categóricos
enteros en lugar de embeddings precalculados.

### `models_ruleid.py`
Define las variantes categóricas de los modelos, con una capa
`nn.Embedding` entrenable como front-end en lugar de recibir
embeddings BERT:
- `TimeAwareTransformerLogKey`
- `DeepLogBaselineLogKey`
- `LogFormerAdapterLogKey`

Reutiliza `TrainingMixin` y `LogAttentionEncoderLayer` de `models.py`.

---

## 4. Scripts de entrenamiento y experimentación

### `run_experiments.py`
Entrena y evalúa los modelos de un solo paso (`transformer`, `bert`,
`deeplog`) sobre un dataset ya preprocesado. Guarda métricas, tiempos
de entrenamiento/inferencia y el estado del mejor modelo.

```bash
python run_experiments.py \
    --dataset ./data/windows.pkl \
    --output_dir ./results \
    --device mps \
    --models transformer,bert,deeplog \
    --max_seq_len 20
```

### `run_logformer.py`
Entrena `LogFormerAdapter` en sus dos etapas (pre-entrenamiento con
encoder completo, luego adapter tuning con encoder congelado) sobre
representación rica.

```bash
python run_logformer.py \
    --dataset ./data/windows.pkl \
    --output_dir ./results \
    --device mps
```

### `run_experiments_ruleid.py`
Análogo a `run_experiments.py` para los modelos con representación
categórica (`transformer_logkey`, `deeplog_logkey`).

### `run_logformer_ruleid.py`
Análogo a `run_logformer.py` para `LogFormerAdapterLogKey`.

---

## 5. Baselines clásicos (sin redes neuronales)

### `run_baselines.py`
Entrena y evalúa PCA, Isolation Forest y One-Class SVM en modo
semi-supervisado clásico (solo ventanas normales de train) usando el
embedding promedio de BERT como representación.

### `run_baselines_ruleid.py`
Variante con el vector de conteo de log keys ("message count vector")
de Xu et al., con la opción `--tfidf` para la ponderación TF-IDF
propuesta por los mismos autores.

### `run_baselines.sh`
Ejecuta las tres variantes en orden: BERT embedding, count vector
crudo y count vector + TF-IDF (Tabla de baselines clásicos de la
tesis).

```bash
./run_baselines.sh
```

---

## 6. Búsqueda de hiperparámetros

### `grid_search.py`
Búsqueda por grilla con selección por F1 de **validación** (semilla
fija por corrida) y evaluación de test una única vez por modelo, solo
para la configuración ganadora. Genera `grid_<modelo>.csv` (todas las
corridas), `grid_summary.json` (ganadoras + test) y `grid_tables.tex`
(tablas LaTeX listas para el documento). Tres modos:

```bash
# Grilla principal: LR x batch x hidden dim, modelos transformer,
# deeplog y bert (78 configs, ~75 min en M5 Pro/MPS)
./grid_search.sh

# Grilla arquitectónica: num_heads y num_layers con el resto fijo
# en la configuración común; solo transformer y deeplog (~8 min).
# bert no participa: su arquitectura (12 capas, 12 cabezas) está
# fijada por el preentrenamiento.
./grid_arch.sh

# Reevaluación: reentrena solo las configs ganadoras de un
# grid_summary.json existente y rehace el test final, sin repetir
# la grilla
python grid_search.py --dataset data/windows.pkl --device mps \
    --output_dir results/grid --from_summary results/grid/grid_summary.json
```

Nota: la columna `best_epoch` de los CSV no es confiable (queda en 0
en la mayoría de las corridas); usar `n_epochs_run`.

### `regenerar_tablas.py`
Regenera `grid_tables.tex` a partir de los CSV y el JSON existentes,
sin reentrenar.

```bash
python regenerar_tablas.py --grid_dir results/grid
```

---

## 7. Análisis y visualización de resultados

### `eda_soc_logs.py`
Análisis exploratorio de datos sobre los logs normales e incidente:
balance de clases, distribución de eventos por ventana, top log keys,
volumen en el tiempo, tácticas MITRE, niveles de alerta, inter-arrival
time. Genera tablas CSV y figuras PNG.

```bash
python eda_soc_logs.py \
    --normal <carpeta_normal> --incidente <carpeta_incidente> \
    --salida ./eda_output
```

Incluye el cálculo explícito de percentiles (P90/P95/P99) de eventos
por ventana, por clase y combinado — los valores citados en la
justificación de `max_seq_len` de la tesis. `eda_task_scheduler.sh`
y `eda_pfsense.sh` lo ejecutan con las rutas de cada experimento
(sistema Windows y pfSense respectivamente).

### `estimar_submuestreo.py`
Modelo de capacidad de memoria: estima el máximo de eventos
procesables según la RAM disponible (bytes/evento medidos desde un
`.pkl` real o calculados teóricamente) y el submuestreo mínimo
resultante. Respalda la sección "Estimación del submuestreo mínimo
aplicable" de la tesis.

```bash
python estimar_submuestreo.py \
    --pkl data/windows_pfsense.pkl --eventos-pkl 100000 \
    --eventos-total 6125736 --ram-gb 24
```

### `estimate_capacity.py`
Herramienta de planificación de capacidad más general: combina
mediciones de disco, RAM y throughput para estimar volúmenes y
tiempos, con puntos de calibración de `memory_profile.py`.

### `explorar_sistemas.py`
Script de exploración sobre los archivos JSON crudos (comprimidos o
no) de Wazuh: cuenta agentes, ubicaciones y organizaciones distintas
sin cargar todo en memoria. Se usó para relevar la composición real
del dataset (Tablas de sistemas por organización en el capítulo de
Desarrollo).

```bash
python explorar_sistemas.py --carpeta <carpeta_raw> --limite 500000
```

### `parse_training_log.py`
Parsea los logs de consola de `run_experiments.py` / `run_logformer.py`
(con formato `Época N — Loss: ... | Val Loss: ... | ...`) y genera:
CSVs por modelo con las métricas por época, y las figuras de curvas
comparativas (`val_f1_comparison.png`, `val_loss_comparison.png`,
`train_loss_comparison.png`, `test_comparison_barplot.png` con eje Y
dinámico, y el detalle individual por modelo).

```bash
python parse_training_log.py \
    --log results/training_full.log \
    --output_dir results/curves
```

### `parse_ruleid.py`
Variante de `parse_training_log.py` con colores/etiquetas propios
para los modelos de representación categórica (`*_logkey`).

### `plot_ruleid.py`
Genera los gráficos de barras comparativos leyendo directamente los
JSON de resultados (no valores hardcodeados), combinando representación
rica y categórica en un único gráfico o por separado.

```bash
python plot_ruleid.py \
    --results_dir ./results \
    --results_ruleid_dir ./results_ruleid \
    --output_dir ./results/curves
```

---

## Flujo típico de un experimento completo

En la práctica, cada paso del pipeline se ejecuta mediante un script
`.sh` que fija los parámetros usados y redirige la salida a un log.
Todos los `.sh` viven en la raíz del proyecto, junto a los `.py`
correspondientes (no se usan subcarpetas).

### Sistema Windows (task scheduler) — representación rica

```bash
./extract_task_scheduler.sh   # extracción jq+parallel desde raw
./prepare_dataset.sh          # genera ./data/windows.pkl
./run_experiments.sh          # transformer, bert, deeplog -> results/training_v2.log
./run_logformer.sh            # LogFormer 2 etapas -> results/training_logformer_v2.log
./combine_and_compare.sh      # combina logs y genera figuras en results/curves
./run_baselines.sh            # baselines clásicos (PCA, iForest, OC-SVM)
./grid_search.sh              # búsqueda de hiperparámetros (LR, batch, hidden)
./grid_arch.sh                # (opcional) búsqueda arquitectónica (heads, layers)
```

### Sistema Windows — representación categórica (rule_id)

```bash
./prepare_dataset_ruleid.sh          # genera ./data/windows_ruleid.pkl
./run_experiments_ruleid.sh          # transformer_logkey, deeplog_logkey
./run_logformer_ruleid.sh            # LogFormer-LogKey 2 etapas
./parse_ruleid.sh                    # combina logs, genera curvas y barplots
                                      # (incluye comparación con representación rica)
```

### Generalización a pfSense

```bash
./extract_pfsense.sh                 # extracción jq+parallel desde raw
./prepare_dataset_pfsense.sh         # genera ./data/windows_pfsense_500k.pkl
                                      # (ajustar --max_events_per_class según
                                      #  el volumen deseado; ver Sección de
                                      #  generalización de la tesis para el
                                      #  historial de valores probados:
                                      #  50k, 150k, 500k por clase)
./run_experiments_pfsense.sh         # transformer, bert, deeplog, logformer
                                      # (ajustar --max_seq_len: 20 en la
                                      #  primera iteración, 64 en la corregida)
./parse_training_log_pfsense.sh      # combina logs y genera figuras
```

### Diagnóstico

```bash
python check_seqlen.py <path_al_pkl>   # distribución de eventos por ventana,
                                        # usado para diagnosticar el
                                        # truncamiento en pfSense
```

---

## Scripts `.sh` — referencia rápida

| Script | Descripción |
|---|---|
| `prepare_dataset.sh` | Preprocesa el sistema Windows (representación rica) |
| `prepare_dataset_ruleid.sh` | Preprocesa el sistema Windows (representación categórica) |
| `prepare_dataset_pfsense.sh` | Preprocesa pfSense con submuestreo configurable |
| `extract_task_scheduler.sh` | Extracción `jq`+`parallel` de agente 256 (Windows) desde raw |
| `extract_pfsense.sh` | Extracción `jq`+`parallel` de agente 016 desde raw |
| `run_experiments.sh` | Entrena transformer/bert/deeplog (Windows, rep. rica) |
| `run_experiments_ruleid.sh` | Entrena transformer/deeplog categóricos |
| `run_experiments_pfsense.sh` | Entrena los 4 modelos sobre pfSense |
| `run_logformer.sh` | Entrena LogFormer 2 etapas (Windows, rep. rica) |
| `run_logformer_ruleid.sh` | Entrena LogFormer-LogKey 2 etapas |
| `combine_and_compare.sh` | Combina logs de Windows (rep. rica) y grafica |
| `parse_ruleid.sh` | Combina logs categóricos y grafica (incl. comparación) |
| `parse_training_log_pfsense.sh` | Combina logs de pfSense y grafica |
| `run_baselines.sh` | Baselines clásicos: BERT emb., count vector, TF-IDF |
| `grid_search.sh` | Grilla principal de hiperparámetros (3 modelos) |
| `grid_arch.sh` | Grilla arquitectónica (heads/layers, sin bert) |
| `eda_task_scheduler.sh` | EDA sobre los logs de Windows (task scheduler) |
| `eda_pfsense.sh` | EDA sobre los logs de pfSense |

**Nota:** los nombres de archivo de log y de dataset (`_v2`, `_500k`,
`_fixed`, etc.) reflejan el historial de iteraciones del proyecto
tal como se ejecutaron. Antes de reutilizar un `.sh`, verificar que
las rutas de `--dataset`, `--output_dir` y los nombres de log dentro
del script correspondan a la corrida que se desea reproducir —en
particular, `run_experiments_pfsense.sh` documenta en comentarios
los dos valores de `--max_seq_len` (20 y 64) usados en las dos
iteraciones del experimento de pfSense descritas en la tesis.

---

## Notas de reproducibilidad

- Todos los scripts de entrenamiento guardan `max_seq_len` en el JSON
  de resultados desde la corrección aplicada en julio 2026.
- Los datasets `.pkl` generados por `prepare_dataset*.py` incluyen
  metadatos de tiempo de generación, tamaño de ventana, modelo BERT
  usado y si se aplicó submuestreo (`max_events_per_class`).
- El split de datos es determinístico dado el mismo `seed` (por
  defecto 42) en `stratified_split_by_origin`.