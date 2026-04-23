import torch


class RealTimeConfig:
    # Ventanas de tiempo
    window_minutes = 5          # Ventana de 5 minutos
    step_minutes = 1            # Paso de 1 minuto
    chunk_minutes = 360         # Chunks de 6 horas (360 minutos)
    max_events_per_chunk = 50000

    # Modelo
    embedding_dim = 768
    hidden_dim = 256
    num_heads = 8
    num_layers = 2
    use_sequence_embeddings = True  # Usar secuencia completa en lugar de promedio
    max_seq_len = 100  # Longitud máxima de secuencia (si se usa)
   

    # Entrenamiento
    batch_size = 32         # Con 15GB GPU podemos usar batch grande
    learning_rate = 2e-5
    num_epochs = 50
    alpha = 0.1
    
    # Early stopping
    patience = 7
    min_delta = 0.001
    mode = 'max'


    # Hardware
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Paths
    normal_logs_path = '../LogBert/data/processed/Legitimos/task_scheduler_ASESP/'
    incident_logs_path = '../LogBert/data/processed/Incidentes/task_scheduler_ASESP/'
    model_save_path = f'modelos/task_scheduler_ASESP/'