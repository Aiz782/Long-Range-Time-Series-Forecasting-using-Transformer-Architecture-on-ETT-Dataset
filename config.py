import torch

class Config:
    DATA_PATH = "ETTh1.csv"   # Change path
    TARGET_COL = "OT"
    
    INPUT_WINDOW = 96
    OUTPUT_WINDOW = 24
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    D_MODEL = 64
    N_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.1
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
