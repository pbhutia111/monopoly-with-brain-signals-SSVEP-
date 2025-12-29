#!/usr/bin/env python3
"""
SSVEP Brain-Controlled Monopoly using OpenBCI Ganglion + PsychoPy.
CCA-based frequency detection with bandpass preprocessing.

Install:
  pip install brainflow psychopy numpy scipy matplotlib pandas scikit-learn pyautogui

Example:
  python ssvep_experiment.py run \
      --serial "/dev/tty.usbserial-XXXX" \
      --freqs 7 13 17 --trial-sec 10 --rest-sec 4 --reps 2 \
      --outfile recording.csv --first-4-channels
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq

# BrainFlow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# PsychoPy
from psychopy import visual, core, event, monitors

import pyautogui
import os
import time


def move_and_click(x, y, delay=0.5):
  print(f"[DEBUG] Moving to ({x}, {y}) and clicking")
  pyautogui.moveTo(x, y)
  time.sleep(0.5)
  pyautogui.click()
  time.sleep(0.2)
  pyautogui.click()

def start_game():
  move_and_click(1235, 826, delay=1.5)  # Click START GAME

def roll_dice():
  move_and_click(568, 467, delay=1)  # Click ROLL DICE
   
def ok():
  move_and_click(475, 609, delay=1)  # Click OK (need to update)

def buy():
  move_and_click(348, 665, delay=1)  # Click BUY
    
def pay_jail(): # pay 20 or roll for doubles
  move_and_click(324, 609, delay=1)  # Click PAY 50
    
def bid():
  move_and_click(630, 565, delay=1)  # Click BID
  
def fold():
  move_and_click(630, 620, delay=1)  # Click FOLD
    
def reject():
  move_and_click(630, 620, delay=1)  # Click REJECT
  
def accept():
  move_and_click(630, 565, delay=1)
  
def end_turn():
  move_and_click(540, 420, delay=0.5)  # Currently unused
  
def focus_arc():
  print("[DEBUG] Focusing Arc browser")
  os.system('osascript -e \'tell application "Arc" to activate\'')
  
def close():
  move_and_click(535, 552, delay=1)  # Click CLOSE

def now_s():
  return time.time()


# =======================
# Stimulus / Flicker UI
# =======================
@dataclass
class FlickerTarget:
    label: str
    freq_hz: float
    rect: visual.Rect
    phase: float = 0.0
    duty: float = 0.5

    def update_visibility(self, t_s: float):
        frac = (self.phase + self.freq_hz * t_s) % 1.0
        self.rect.opacity = 1.0 if frac < self.duty else 0.0


def build_targets(win, freqs: List[float]) -> List[FlickerTarget]:
    """
    Place flicker boxes in corners based on number of frequencies:
    - 2 freqs: Top-Left, Bottom-Right
    - 3 freqs: Top-Left, Top-Right, Bottom-Left
    - 4 freqs: All four corners (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    """
    n = len(freqs)
    
    # Box size and corner offset
    box_size = 0.25  # Size of each flicker box
    offset = 0.65    # Distance from center (0,0) to corner
    
    # Define corner positions: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
    corner_positions = [
        (-offset, offset),   # Top-Left
        (offset, offset),    # Top-Right
        (-offset, -offset),  # Bottom-Left
        (offset, -offset)    # Bottom-Right
    ]
    
    # Position mapping based on number of frequencies
    if n == 2:
        # Top-Left and Bottom-Right (diagonal)
        positions = [corner_positions[0], corner_positions[3]]
    elif n == 3:
        # Top-Left, Top-Right, Bottom-Left
        positions = [corner_positions[0], corner_positions[1], corner_positions[2]]
    elif n == 4:
        # All four corners
        positions = corner_positions
    else:
        # Fallback: use as many corners as needed
        positions = corner_positions[:n]
    
    targets = []
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    
    for i, f in enumerate(freqs):
        pos = positions[i]
        color = "white"  # Default color for all boxes
        if f == 7:
            color = "red"
        elif f == 13:
            color = "blue"
        elif f == 17:
            color = "green"
        
        # Create flicker rectangle
        rect = visual.Rect(
            win, 
            width=box_size, 
            height=box_size, 
            fillColor=color, 
            lineColor=color,
            pos=pos, 
            opacity=0.0
        )
        
        # Create label below the box
        label_offset = 0.15  # Distance below box
        lbl = visual.TextStim(
            win, 
            text=f"{f:.1f} Hz\n{corner_labels[i] if i < len(positions) else ''}",
            pos=(pos[0], pos[1] - label_offset), 
            height=0.05, 
            color='white',
            alignText='center'
        )
        
        targets.append(FlickerTarget(
            label=f"{f:.1f} Hz ({corner_labels[i] if i < len(positions) else ''})", 
            freq_hz=f, 
            rect=rect
        ))
        
        # Draw the label (it stays visible)
        lbl.draw()
    
    # Add a center fixation cross for reference
    fixation = visual.TextStim(
        win,
        text='+',
        height=0.1,
        color='gray',
        pos=(0, 0)
    )
    fixation.draw()
    
    win.flip()
    return targets


# =======================
# BrainFlow Recorder - Ganglion Only
# =======================
class BFRecorder:
    def __init__(self, serial_port: str, first_4_channels: bool = False):
        self.params = BrainFlowInputParams()
        
        if not serial_port:
            raise ValueError("--serial is required for Ganglion board")
        
        self.params.serial_port = serial_port
        self.board_id = BoardIds.GANGLION_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        self.first_4 = first_4_channels

    def start(self, buffer_sec: int = 300, streamer_params: str = ""):
        self.board.prepare_session()
        self.board.start_stream(num_samples=int(buffer_sec * self.get_sr()),
                                streamer_params=streamer_params)

    def insert_marker(self, value: float):
        try:
            self.board.insert_marker(value)
        except Exception:
            pass  # marker not supported -> ignore

    def stop_and_get(self) -> np.ndarray:
        self.board.stop_stream()
        data = self.board.get_board_data()  # (num_channels, num_samples)
        self.board.release_session()
        return data

    def get_sr(self) -> int:
        return BoardShim.get_sampling_rate(self.board_id)

    def get_eeg_channel_indices(self) -> List[int]:
        eeg = BoardShim.get_eeg_channels(self.board_id)
        if self.first_4:
            return eeg[:4]
        return eeg

    def get_ts_channel_index(self) -> int:
        return BoardShim.get_timestamp_channel(self.board_id)

    def get_marker_channel_index(self) -> Optional[int]:
        try:
            return BoardShim.get_marker_channel(self.board_id)
        except Exception:
            return None


# =======================
# Run Experiment
# =======================
def run_block(win: visual.Window, targets: List[FlickerTarget], block_sec: float, recorder: BFRecorder):
    recorder.insert_marker(1)
    clock = core.Clock()
    clock.reset()

    info = visual.TextStim(win, text=f"Focus on all targets\n(ESC to stop)",
                           pos=(0, 0.6), height=0.05, color='white')

    while True:
        t = clock.getTime()
        if t >= block_sec:
            break
        
        info.draw()
        for target in targets:
            target.update_visibility(t_s=t)
            target.rect.draw()
        win.flip()
    core.wait(0.1)


def run_experiment_with_window(win: visual.Window, serial_port: str, 
                                freqs: List[float], trial_sec: float, rest_sec: float, 
                                reps: int, outfile: str, first_4_channels: bool):
    """Run SSVEP experiment with Ganglion board."""
    actual_hz = win.getActualFrameRate(
        nIdentical=20, nMaxFrames=120, nWarmUpFrames=60, threshold=1) or 60.0
    print(f"[INFO] Measured display refresh ~{actual_hz:.2f} Hz")

    intro = visual.TextStim(
        win,
        text=(f"SSVEP Ganglion\n\nTargets: {', '.join([f'{f:.1f} Hz' for f in freqs])}\n"
              f"Trial {trial_sec:.1f}s | Rest {rest_sec:.1f}s | Reps {reps}\n\n"),
        color='white', height=0.06
    )
    intro.draw()
    win.flip()

    targets = build_targets(win, freqs)

    recorder = BFRecorder(serial_port=serial_port,
                          first_4_channels=first_4_channels)
    
    sr = recorder.get_sr()
    print(f"[INFO] Ganglion board streaming at {sr} Hz")

    ret_data = []
    try:
        for r in range(reps):
            recorder.start(buffer_sec=int(len(freqs) * (trial_sec + rest_sec) + 60))

            print(f"[RUN] rep {r+1}/{reps} all targets")
            run_block(win, targets, trial_sec, recorder)
            rest_txt = visual.TextStim(
                win, text="Rest", color='white', height=0.07)
            rest_txt.draw()
            win.flip()
            core.wait(rest_sec)
            data = recorder.stop_and_get()
            eeg_idxs = recorder.get_eeg_channel_indices()
            ts_idx = recorder.get_ts_channel_index()
            marker_idx = recorder.get_marker_channel_index()

            out = {
                "timestamps": data[ts_idx, :],
                "marker": data[marker_idx, :] if marker_idx is not None else np.zeros(data.shape[1]),
            }

            for k, ch in enumerate(eeg_idxs, start=1):
                out[f"EEG{k}"] = data[ch, :]

            df = pd.DataFrame(out)
            df.to_csv(f"{r}-"+outfile, index=False)
            print(f"[SAVE] {outfile}  shape={df.shape}")
            print(r)

            ret_data.append(df)
            
    except KeyboardInterrupt:
        print("[WARN] Aborted by user.")
    except Exception as e:
        print("ERROR!: ", e)
        raise e
    
    return ret_data



# =======================
# Analysis + Preprocessing
# =======================
def butter_bandpass(low, high, fs, order=4):
    ny = fs / 2.0
    b, a = butter(order, [low / ny, high / ny], btype='band')
    return b, a


def apply_bandpass(x, fs, low, high, order=4):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)


def infer_sr_from_timestamps(ts: np.ndarray) -> float:
    return 200


def find_marker_indices(marker: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (index, value) pairs where a positive marker was inserted.
    BrainFlow marker appears as an impulse at the insertion sample.
    """
    idxs = np.where(marker > 0)[0]
    vals = marker[idxs].astype(int) if len(idxs) else np.array([], dtype=int)
    return list(zip(idxs.tolist(), vals.tolist()))


def epoch_blocks(
    data: np.ndarray,        # shape (n_ch, n_samp)
    sr: float,
    marker: np.ndarray,      # shape (n_samp,)
    trial_sec: float,
    drop_sec: float
) -> Dict[int, List[np.ndarray]]:
    """
    Build epochs per target index using marker impulses.
    Returns dict: target_id -> list of arrays shaped (n_ch, n_samples_per_epoch)
    """
    n = data.shape[1]
    samples_trial = int(trial_sec * sr)
    samples_drop = int(drop_sec * sr)

    epochs_by_target: Dict[int, List[np.ndarray]] = {}
    events = find_marker_indices(marker)
    for idx, val in events:
        start = idx + samples_drop
        end = idx + samples_trial
        if end <= n and start < end:
            seg = data[:, start:end].copy()
            epochs_by_target.setdefault(val, []).append(seg)
    return epochs_by_target


def compute_psd_mean(ch_data: np.ndarray, fs: float, welch_sec: float = 2.0):
    nperseg = max(int(welch_sec * fs), 128)
    noverlap = nperseg // 2
    f, pxx = welch(ch_data, fs=fs, nperseg=nperseg,
                   noverlap=noverlap, detrend="constant")
    return f, pxx


# =======================
# CCA-based SSVEP Classifier
# =======================
def make_reference_signals(freq, t, n_harmonics=2):
    """
    Build reference signals for a single freq.
    Returns X of shape (n_samples, 2*n_harmonics)
    (cos and sin for each harmonic 1..n_harmonics).
    """
    refs = []
    for h in range(1, n_harmonics+1):
        refs.append(np.cos(2*np.pi*(h*freq)*t))
        refs.append(np.sin(2*np.pi*(h*freq)*t))
    X = np.column_stack(refs)
    return X


def cca_ssvep_predictor(Y, t=None, fs=200, freqs=(7,10,12),
                        n_harmonics=2, preprocess=True):
    """
    Y: (n_samples, n_channels)
    t: time vector in seconds OR
    fs: sampling frequency (Hz) if t is None
    freqs: iterable of frequencies to test
    n_harmonics: number of harmonics to include in reference
    preprocess: True -> demean and z-score channels (recommended)
    Returns:
      results: dict mapping freq -> dict with keys
        'corr' : canonical correlation (float)
        'Y_c'  : EEG canonical variate (n_samples,) — predicted timecourse from EEG
        'X_c'  : reference canonical variate (n_samples,) — matched reference timecourse
        'wx'   : projection weights for X (shape (n_ref,))
        'wy'   : projection weights for Y (shape (n_channels,))
    """
    if t is None:
        if fs is None:
            raise ValueError("Either t or fs must be provided")
        n_samples = Y.shape[0]
        t = np.arange(n_samples) / float(fs)
    else:
        t = np.asarray(t)
    
    Y = np.asarray(Y)
    n_samples, n_channels = Y.shape
    
    if preprocess:
        Y = (Y - Y.mean(axis=0, keepdims=True))
        # z-score channels to equalize variances (optional)
        stds = Y.std(axis=0, ddof=0)
        stds[stds==0] = 1.0
        Y = Y / stds
    
    results = {}
    for f in freqs:
        X = make_reference_signals(f, t, n_harmonics=n_harmonics)  # (n_samples, n_ref)
        # optionally demean X as well
        X = X - X.mean(axis=0, keepdims=True)
        
        # run 1-component CCA
        cca = CCA(n_components=1, max_iter=500)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)  # both are (n_samples, 1)
        
        xvar = X_c[:,0]
        yvar = Y_c[:,0]
        
        # canonical correlation
        corr = np.corrcoef(xvar, yvar)[0,1]
        
        # get projection weights (components)
        # In sklearn, cca.x_weights_ and cca.y_weights_
        wx = cca.x_weights_[:,0]  # shape (n_ref,)
        wy = cca.y_weights_[:,0]  # shape (n_channels,)
        
        results[f] = {
            'corr': float(np.abs(corr)),   # abs correlation is used for detection
            'corr_signed': float(corr),
            'Y_c': yvar,
            'X_c': xvar,
            'wx': wx,
            'wy': wy
        }
    
    return results



def analyze_psd(
    df: pd.DataFrame,
    freqs: List[float],
    fmin: float,
    fmax: float,
    trial_sec: float,
    drop_sec: float,
    bp_low: float,
    bp_high: float,
    notch: Optional[float],
    per_target: bool,
    welch_sec: float = 2.0
):
    """
    CCA-based SSVEP classifier.
    Filters EEG, runs sliding windows, applies CCA for each window,
    then votes across windows to pick best frequency.
    Returns "a" (7 Hz), "b" (13 Hz), or "c" (17 Hz).
    """
    eeg_channels = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    eeg_data = df[eeg_channels].values  # Shape: (n_samples, 4)

    # ganglion sr is 200 Hz
    sr = 200

    def bandpass_filter(data, lowcut, highcut, fs):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def bandstop_filter(data, lowcut, highcut, fs):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='bandstop')
        return filtfilt(b, a, data, axis=0)

    # Preprocessing: Bandpass filter + Notch filter for powerline noise
    eeg_filtered = bandpass_filter(eeg_data, bp_low, bp_high, sr)
    eeg_filtered = bandstop_filter(eeg_filtered, 58, 62, sr)  # 60 Hz notch
    print(f"Filtered EEG shape: {eeg_filtered.shape}")

    # Sliding windows
    window_size = 600  # 3 seconds at 200 Hz
    step_size = 100    # 0.5 seconds overlap
    windows = []
    for start in range(0, len(eeg_filtered) - window_size, step_size):
        end = start + window_size
        window = eeg_filtered[start:end, :]
        windows.append(window)

    if len(windows) == 0:
        print("[WARN] No windows extracted; returning default 'a'")
        return "a"

    print(f"Created {len(windows)} windows of shape {windows[0].shape}")

    # Run CCA on each window and collect votes
    target_freqs = freqs  # use the passed frequencies: [7.0, 9.5, 12.0]
    freq_vote_counts = {f: 0 for f in target_freqs}

    all_corrs = {f: [] for f in target_freqs}
    for window in windows:
        # Y shape: (n_samples, n_channels)
        res = cca_ssvep_predictor(
            Y=window,
            fs=sr,
            freqs=target_freqs,
            n_harmonics=2,
            preprocess=True
        )
        # Pick best frequency for this window
        best_f = max(res.keys(), key=lambda f: res[f]['corr'])
        freq_vote_counts[best_f] += 1
        for f in target_freqs:
            all_corrs[f].append(res[f]['corr'])

    # Print average correlations for each frequency
    print(f"CCA vote counts: {freq_vote_counts}")
    print("Avg correlations:", {f: f"{np.mean(all_corrs[f]):.4f}" for f in target_freqs})

    # Majority vote
    best_freq = max(freq_vote_counts.keys(), key=lambda f: freq_vote_counts[f])
    print(f"Best frequency by vote: {best_freq} Hz")

    # Map to "a", "b", "c" based on frequency index
    # freqs[0] -> "a", freqs[1] -> "b", freqs[2] -> "c"
    freq_to_letter = {freqs[i]: chr(ord('a') + i) for i in range(len(freqs))}
    return freq_to_letter.get(best_freq, "a")


# =======================
# CLI
# =======================
def main():
    parser = argparse.ArgumentParser(
        description="SSVEP Brain-Controlled Monopoly with OpenBCI Ganglion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Run
    pr = sub.add_parser("run", help="Run SSVEP experiment with Ganglion board")
    pr.add_argument("--serial", type=str, required=True,
                    help="Serial port for Ganglion (e.g., /dev/tty.usbserial-XXXX)")
    pr.add_argument("--freqs", type=float, nargs="+",
                    default=[7.0,13.0,17.0], help="Target flicker frequencies (Hz)")
    pr.add_argument("--trial-sec", type=float, default=10.0,
                    help="Seconds per target block")
    pr.add_argument("--rest-sec", type=float, default=5.0,
                    help="Seconds rest between blocks")
    pr.add_argument("--reps", type=int, default=3,
                    help="How many times to cycle through targets")
    pr.add_argument("--outfile", type=str,
                    default="recording.csv", help="CSV output path")
    pr.add_argument("--first-4-channels", action="store_true",
                    help="Use only the first 4 EEG channels in output")

    args = parser.parse_args()
    mon = monitors.Monitor('testMonitor')
    win = visual.Window(size=(1800,1000), fullscr=False, screen=0, color='black', units='norm', monitor=mon)
    

    command = ""
    for i in range(0,10):
        print("Trial: ", i)
        data_lst = run_experiment_with_window(
            win=win,
            serial_port=args.serial,
            freqs=args.freqs,
            trial_sec=args.trial_sec,
            rest_sec=4,
            reps=2,
            outfile=args.outfile,
            first_4_channels=args.first_4_channels,
            
        )
        print(f"{'X'*90}")

        hzO = analyze_psd(
            df=data_lst[0],
            freqs=args.freqs,
            fmin=(args.freqs[0] - 6),
            fmax=(args.freqs[-1] + 10),
            trial_sec=args.trial_sec,
            drop_sec=0.5,
            bp_low=(args.freqs[0] - 1.5),
            bp_high=(args.freqs[-1] + 5),
            notch=60,
            per_target=False,
            welch_sec=2.0
        )
        hz1 = analyze_psd(
            df=data_lst[1],
            freqs=args.freqs,
            fmin=(args.freqs[0] - 6),
            fmax=(args.freqs[-1] + 10),
            trial_sec=args.trial_sec,
            drop_sec=0.5,
            bp_low=(args.freqs[0] - 1.5),
            bp_high=(args.freqs[-1] + 5),
            notch=60,
            per_target=False,
            welch_sec=2.0
        )
        if hzO != None and hz1 != None:
            command = f"{hzO}{hz1}"
        
        print("COMMAND: ", command)
        match command:
            case "aa":
                print("aa")
                start_game()
                time.sleep(1.0)
                close()
            case "ab":
                print("ab")
                roll_dice()
            case "ac":
                print("ac")
                ok()
            case "ba":

                buy()
                print("ba")
            case "bb":
                close()
                print("bb")
                
            case "bc":
                bid()
                print("bc")
                
            case "ca":
                fold()
                print("ca")
                
            case "cb":
                reject()
                print("cb")
                
            case "cc":
                accept()
                print("cc")
                
            


if __name__ == "__main__":
    main()
tha