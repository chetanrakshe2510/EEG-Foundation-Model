# data_loader.py
import pandas as pd
from tqdm import tqdm
import mne_bids
from braindecode.datasets import BaseDataset, BaseConcatDataset

from config import Config

def load_bids_data(releases: list, task: str):
    """
    Loads multiple BIDS releases for a specific task into a BaseConcatDataset.
    
    Args:
        releases (list): List of release names to load (e.g., ["R1", "R2"]).
        task (str): The BIDS task identifier (e.g., 'contrastChangeDetection').

    Returns:
        BaseConcatDataset: A concatenated dataset of all found recordings.
    """
    all_datasets = []
    print(f"Loading data for task '{task}' from releases: {releases}")

    for release_name in releases:
        folder_name = Config.RELEASE_TO_FOLDER.get(release_name)
        if not folder_name:
            print(f"Warning: Folder for release {release_name} not found in config. Skipping.")
            continue
            
        bids_root = Config.DATA_ROOT / folder_name
        if not bids_root.exists():
            print(f"Warning: Directory not found for release {release_name} at {bids_root}. Skipping.")
            continue

        participants_df = pd.read_csv(bids_root / "participants.tsv", sep='\t')
        subject_ids = [sid.replace('sub-', '') for sid in participants_df['participant_id']]

        for subject in tqdm(subject_ids, desc=f"Loading {release_name}"):
            # The competition dataset can have up to 3 runs for CCD
            for run in ['1', '2', '3']: 
                try:
                    bids_path = mne_bids.BIDSPath(
                        subject=subject, task=task, run=run,
                        root=bids_root, datatype='eeg', extension='.set'
                    )
                    raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
                    raw.load_data()
                    
                    # Add demographic info to dataset description
                    subject_meta = participants_df.loc[participants_df['participant_id'] == f'sub-{subject}']
                    description = {'subject': subject, 'run': run, 'release': release_name}
                    if not subject_meta.empty:
                        description['age'] = subject_meta['age'].iloc[0]
                        description['sex'] = subject_meta['sex'].iloc[0]

                    ds = BaseDataset(raw, description=description)
                    all_datasets.append(ds)
                except FileNotFoundError:
                    # It's normal for some subjects/runs not to exist, so we pass silently.
                    pass 
                    
    print(f"Loaded {len(all_datasets)} recording sessions for task '{task}'.")
    return BaseConcatDataset(all_datasets)