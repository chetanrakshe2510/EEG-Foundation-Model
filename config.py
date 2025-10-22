from pathlib import Path
import torch
import warnings

# Suppress pandas computation warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.computation.expressions")


class Config:
    # ==========================================================================
    # --- PATHS ---
    # ==========================================================================
    DATA_ROOT = Path(r"G:\TA_work\EEG Competition\Modular_code\data")  # Main directory for all BIDS datasets

    TRAIN_RELEASES = ["R_train"]
    VALID_RELEASE = "R_test"

    RELEASE_TO_FOLDER = {
        "R1": "ds005505", "R2": "ds005506", "R3": "ds005507",
        "R4": "ds005508", "R5": "ds005509", "R6": "ds005510",
        "R7": "ds005511", "R8": "ds005512", "R9": "ds005514",
        "R10": "ds005515", "R11": "ds005516",
        "R_train": "ds_train", "R_test": "ds_test",
    }

    # Directory for saving outputs (plots, models, logs)
    SAVE_PATH = Path("artifacts")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # --- TASK DEFINITIONS ---
    # ==========================================================================
    PRETRAIN_TASK = "surroundSupp"          # Challenge 1 pre-training task
    FINETUNE_TASK = "contrastChangeDetection"  # Challenge 1 fine-tuning task

    # ==========================================================================
    # --- PREPROCESSING PARAMETERS ---
    # ==========================================================================
    SFREQ = 100
    L_FREQ = 0.5
    H_FREQ = 40.0           # Should stay below Nyquist frequency (SFREQ / 2)
    ICA_N_COMPONENTS = 20
    ICA_RANDOM_STATE = 42

    # ==========================================================================
    # --- WINDOWING / EPOCHING ---
    # ==========================================================================
    EPOCH_LEN_SECONDS = 2.0
    ANCHOR_EVENT = "stimulus_anchor"  # Anchor event for Challenge 1
    TMIN = 0.5
    TMAX = 2.5

    # ==========================================================================
    # --- MODEL & TRAINING HYPERPARAMETERS ---
    # ==========================================================================
    MODEL_NAME = "EEGNeX"
    LEARNING_RATE = 3e-3
    WEIGHT_DECAY = 0
    BATCH_SIZE = 256
    NUM_WORKERS = 0  # Set > 0 if you have a multi-core CPU

    N_EPOCHS_PRETRAIN = 30  # Epochs for self-supervised pre-training
    N_EPOCHS_FINETUNE = 60  # Epochs for fine-tuning

    # ==========================================================================
    # --- VALIDATION & EARLY STOPPING ---
    # ==========================================================================
    VALID_FRAC = 0.2
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 1e-4

    # ==========================================================================
    # --- SYSTEM SETTINGS ---
    # ==========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2025

    # ==========================================================================
    # --- OPTIONAL FEATURES ---
    # ==========================================================================
    USE_LR_FINDER = False  # Toggle for LR Finder (non-blocking, saves plot)


# ------------------------------------------------------------------------------
# --- CONFIG ACCESSOR FUNCTION ---
# ------------------------------------------------------------------------------
def get_config(test_mode=False):
    """
    Returns the Config object.
    If test_mode=True, modifies certain parameters for faster testing.
    """
    cfg = Config

    if test_mode:
        cfg.DATA_ROOT = Path("./tests/dummy_bids_data")
        cfg.TRAIN_RELEASES = ["DUMMY"]
        cfg.RELEASE_TO_FOLDER = {"DUMMY": "ds000001"}
        cfg.N_EPOCHS_PRETRAIN = 1
        cfg.N_EPOCHS_FINETUNE = 1
        cfg.BATCH_SIZE = 2
        cfg.SAVE_PATH = Path("./artifacts_test")
        cfg.SAVE_PATH.mkdir(parents=True, exist_ok=True)
        print("--- RUNNING IN TEST MODE ---")

    return cfg
