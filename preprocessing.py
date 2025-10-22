# preprocessing.py (Corrected)
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import Preprocessor
from braindecode.datasets import BaseConcatDataset
from mne_bids import get_bids_path_from_fname

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# build_trial_table, add_aux_anchors, annotate_trials_with_target, and
# keep_only_recordings_with do not need changes. For brevity, they are omitted,
# but your existing versions of these functions are correct.

def add_extras_columns(windows_concat_ds, original_concat_ds, desc="contrast_trial_start"):
    """
    Manually adds specified keys from MNE annotations.extras into the
    windows_dataset's metadata DataFrame. Renames 'target' to 'rt_from_stimulus'.
    """
    # ✨ FIX: Add 'target' to the list of keys to extract from the annotations.
    keys_to_add = ['target', 'correct', 'response_type']

    for win_ds, base_ds in zip(windows_concat_ds.datasets, original_concat_ds.datasets):
        if base_ds.raw.annotations is None:
            continue
        
        trial_mask = (base_ds.raw.annotations.description == desc)
        if not np.any(trial_mask):
            continue

        per_trial_extras = [base_ds.raw.annotations.extras[i] for i in np.where(trial_mask)[0]]
        md = win_ds.metadata.copy()
        
        if "i_window_in_trial" not in md.columns:
            raise RuntimeError("Missing 'i_window_in_trial' in metadata.")

        # This logic correctly maps windows back to their original trial
        is_first_window = (md["i_window_in_trial"].to_numpy() == 0)
        trial_ids = is_first_window.cumsum() - 1

        for key in keys_to_add:
            vals = [per_trial_extras[int(t)].get(key) for t in trial_ids]
            
            # ✨ FIX: Rename the 'target' column to 'rt_from_stimulus' to match what the model expects.
            column_name = 'rt_from_stimulus' if key == 'target' else key
            md[column_name] = vals
        
        win_ds.metadata = md
    return windows_concat_ds

# ... (Your other functions: build_trial_table, annotate_trials_with_target, etc.)
# Your `get_base_preprocessors` function is also correct and does not need changes.
def build_trial_table(events_df: pd.DataFrame, min_trial_duration: float = 1.0) -> pd.DataFrame:
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)
    trials = events_df[events_df["value"].str.contains("trial_start", case=False, na=False)].copy()
    if trials.empty: return pd.DataFrame()
    trials["next_onset"] = trials["onset"].shift(-1)
    end_of_recording = events_df["onset"].max() + 1
    trials.fillna({"next_onset": end_of_recording}, inplace=True)
    trials = trials[trials["next_onset"] - trials["onset"] >= min_trial_duration]
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()
    trials = trials.reset_index(drop=True)
    rows = []
    for _, tr in trials.iterrows():
        start, end = float(tr["onset"]), float(tr["next_onset"])
        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])
        if not np.isnan(stim_onset): resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
        else: resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]
        if resp_block.empty: resp_onset, resp_type, feedback = np.nan, None, None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type = resp_block.iloc[0]["value"]
            feedback = resp_block.iloc[0].get("feedback", None)
        rt_from_stim = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
        rt_from_trial = (resp_onset - start) if not np.isnan(resp_onset) else np.nan
        correct = None
        if isinstance(feedback, str): correct = True if feedback == "smiley_face" else (False if feedback == "sad_face" else None)
        rows.append({"trial_start_onset": start, "trial_stop_onset": end, "stimulus_onset": stim_onset, "response_onset": resp_onset, "rt_from_stimulus": rt_from_stim, "rt_from_trialstart": rt_from_trial, "response_type": resp_type, "correct": correct})
    return pd.DataFrame(rows)

def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0, require_stimulus=True, require_response=True):
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath
    events_df = pd.read_csv(events_file, sep="\t").assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise")).sort_values("onset", kind="mergesort").reset_index(drop=True)
    ann_events, event_dict = mne.events_from_annotations(raw, verbose=False)
    if not event_dict: return raw
    id_to_desc = {v: k for k, v in event_dict.items()}
    ann_events_df = pd.DataFrame(ann_events[:, [0, 2]], columns=['sample', 'value'])
    ann_events_df['onset'] = ann_events_df['sample'] / raw.info['sfreq']
    ann_events_df['value'] = ann_events_df['value'].map(id_to_desc)
    events_df = pd.concat([events_df, ann_events_df[['onset', 'value']]], ignore_index=True).sort_values("onset").reset_index(drop=True)
    trials = build_trial_table(events_df)
    if require_stimulus: trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response: trials = trials[trials["response_onset"].notna()].copy()
    if trials.empty: return raw
    if target_field not in trials.columns: raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)
    onsets = trials["trial_start_onset"].to_numpy(float)
    durations = np.full(len(trials), float(epoch_length))
    descs = ["contrast_trial_start"] * len(trials)
    extras = [{"target": float(v) if not pd.isna(v) else None, "stimulus_onset": s_o, "response_onset": r_o, "correct": c, "response_type": r_t} for v, s_o, r_o, c, r_t in zip(targets, trials["stimulus_onset"], trials["response_onset"], trials["correct"], trials["response_type"])]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descs, orig_time=raw.info["meas_date"], extras=extras), verbose=False)
    return raw

def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann = raw.annotations
    mask = (ann.description == "contrast_trial_start")
    if not np.any(mask): return raw
    stim_onsets, resp_onsets, stim_extras, resp_extras = [], [], [], []
    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        stim_t, resp_t = ex.get("stimulus_onset"), ex.get("response_onset")
        if stim_t is not None and not np.isnan(stim_t): stim_onsets.append(float(stim_t)); stim_extras.append(dict(ex, anchor="stimulus"))
        if resp_t is not None and not np.isnan(resp_t): resp_onsets.append(float(resp_t)); resp_extras.append(dict(ex, anchor="response"))
    if stim_onsets or resp_onsets:
        raw.set_annotations(ann + mne.Annotations(onset=np.array(stim_onsets + resp_onsets, dtype=float), duration=np.zeros(len(stim_onsets) + len(resp_onsets)), description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets), orig_time=raw.info["meas_date"], extras=stim_extras + resp_extras), verbose=False)
    return raw

def keep_only_recordings_with(desc, concat_ds):
    kept = [ds for ds in concat_ds.datasets if np.any(ds.raw.annotations.description == desc)]
    return BaseConcatDataset(kept)

def get_base_preprocessors(sfreq, l_freq, h_freq):
    eeg_channels = [f'E{i}' for i in range(1, 129)]
    def set_channel_info(raw):
        ch_mapping = {ch: 'eeg' for ch in raw.ch_names if ch in eeg_channels}
        raw.set_channel_types(ch_mapping)
        montage = mne.channels.make_standard_montage('biosemi128')
        raw.set_montage(montage, on_missing='ignore')
        return raw
    return [
        Preprocessor(set_channel_info, apply_on_array=False),
        Preprocessor('set_eeg_reference', ref_channels=['Cz']),
        Preprocessor(fn='drop_channels', ch_names=['Cz']),
        Preprocessor('filter', l_freq=l_freq, h_freq=h_freq),
        Preprocessor(fn='resample', sfreq=sfreq),
        Preprocessor(lambda x: x * 1e6),
        Preprocessor(lambda x: np.clip(x, a_min=-500, a_max=500)),
    ]