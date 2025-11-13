#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.fft import fft, fftfreq
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core, event, monitors


def now_s():
    return time.time()

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
        color = ""
        if f == 7:
            color = "red"
        if f == 9.5:
            color = "white"
        if f == 12:
            color = "green"
        # Create flicker rectangle
        rect = visual.Rect(
            win, 
            width=box_size, 
            height=box_size, 
            fillColor="white", 
            lineColor="white",
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
# BrainFlow Recorder
# =======================
class BFRecorder:
    def __init__(self, board: str, serial_port: Optional[str], first_4_channels: bool = False):
        self.params = BrainFlowInputParams()

        b = board.lower()
        if b == "cyton":
            if not serial_port:
                raise ValueError("--serial is required for --board cyton")
            self.params.serial_port = serial_port
            self.board_id = BoardIds.CYTON_BOARD.value
        elif b == "ganglion":
            if not serial_port:
                raise ValueError("--serial is required for --board ganglion")
            self.params.serial_port = serial_port
            self.board_id = BoardIds.GANGLION_BOARD.value
        elif b == "synthetic":
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            raise ValueError("Unknown board. Use --board cyton | synthetic")

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


def run_experiment(board: str, serial_port: Optional[str], freqs: List[float],
                   trial_sec: float, rest_sec: float, reps: int, outfile: str,
                   first_4_channels: bool):
    mon = monitors.Monitor('testMonitor')
    win = visual.Window(size=(1470,800), fullscr=False, color='black', screen=1,units='norm', monitor=mon)
    actual_hz = win.getActualFrameRate(
        nIdentical=20, nMaxFrames=120, nWarmUpFrames=60, threshold=1) or 60.0
    print(f"[INFO] Measured display refresh ~{actual_hz:.2f} Hz")

    intro = visual.TextStim(
        win,
        text=(f"SSVEP ({board})\n\nTargets: {', '.join([f'{f:.1f} Hz' for f in freqs])}\n"
              f"Trial {trial_sec:.1f}s | Rest {rest_sec:.1f}s | Reps {reps}\n\n"),
        color='white', height=0.06
    )
    intro.draw()
    win.flip()
    

    targets = build_targets(win, freqs)

    
    recorder = BFRecorder(board=board, serial_port=serial_port,
                          first_4_channels=first_4_channels)
    
    
    sr = recorder.get_sr()
    print(f"[INFO] Board '{board}' streaming at {sr} Hz")
    total_sec = reps * len(freqs) * (trial_sec + rest_sec) + 60
    recorder.start(buffer_sec=int(total_sec))

    
    try:
        for r in range(reps):

            print(f"[RUN] rep {r+1}/{reps} all targets")
            run_block(win, targets, trial_sec, recorder)
            rest_txt = visual.TextStim(
                win, text="Rest", color='white', height=0.07)
            rest_txt.draw()
            win.flip()
            core.wait(rest_sec)

    except KeyboardInterrupt:
        print("[WARN] Aborted by user.")
    except Exception as e:
        print("ERROR!: ", e)
        raise e
    finally:
        print("RE T 2")
        win.close()

    
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
    df.to_csv(outfile, index=False)
    print(f"[SAVE] {outfile}  shape={df.shape}")
    
    return df


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


def apply_notch(x, fs, notch_hz=60.0, q=30.0):
    # iirnotch: w0 (rad/s) normalized (f0 / (fs/2))
    w0 = notch_hz / (fs / 2.0)
    b, a = iirnotch(w0, Q=q)
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
    bp_low = 5.5
    bp_high = 15
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

    def notch_filter(data, freq, fs):
        nyquist = fs / 2
        b, a = iirnotch(freq / nyquist, 30)
        return filtfilt(b, a, data, axis=0)
    
    def bandstop_filter(data, lowcut, highcut, fs):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='bandstop')
        return filtfilt(b, a, data, axis=0)

    eeg_filtered = bandpass_filter(eeg_data, bp_low, bp_high, sr)
    eeg_filtered = notch_filter(eeg_filtered, 60, sr)
    eeg_filtered = bandstop_filter(eeg_filtered, 10, 11, sr)
    print(eeg_filtered.shape)

    windows = []
    window_size = 200
    step_size = 50
        
    for start in range(0, len(eeg_filtered) - window_size, step_size):
        end = start + window_size
        window = eeg_filtered[start:end, :]
        windows.append(window)

    print(f"Created {len(windows)} windows")
    print(f"Each window shape: {windows[1].shape}")

    n_samples = windows[0].shape[0]
    n_channels = windows[0].shape[1]
    sr = 200

    
    target_freqs = np.arange(3, 20, 0.5)

    # Store power for each frequency across all windows
    freq_powers = {float(freq): [] for freq in target_freqs}
    
    # Process each window
    for window in windows:
        # Compute FFT for each channel in this window
        for ch in range(n_channels):
            channel_data = window[:, ch]
            
            fft_values = fft(channel_data)
            fft_freqs = fftfreq(n_samples, 1/sr)
            
            power_spectrum = np.abs(fft_values[:n_samples//2])
            positive_freqs = fft_freqs[:n_samples//2]
            
            # Extract power at each target frequency
            for target_freq in target_freqs:
                idx = np.argmin(np.abs(positive_freqs - target_freq))
                freq_powers[target_freq].append(power_spectrum[idx])

    # Calculate average power across all windows and channels
    avg_powers = {freq: np.mean(powers) for freq, powers in freq_powers.items()}
        
    freqs = list(avg_powers.keys())
    powers = list(avg_powers.values())
    print("Freqs: ", freqs)
    print("Powers: ", powers)

    sevenhz = avg_powers[7]
    ninehz = avg_powers[9.5]
    twelvehz = avg_powers[12]

    print(sevenhz)
    print(ninehz)
    print(twelvehz)

    if (sevenhz > ninehz and sevenhz > twelvehz):
        return "a"
    elif (ninehz > sevenhz and ninehz > twelvehz):
        return "b"
    else:
        return "c"


# =======================
# CLI
# =======================
def exp():

    command = ""
    for i in range(0,2): # 2 trials
        print("Trial: ", i)
        data = run_experiment(
            board="ganglion",
            serial_port="/dev/cu.usbmodem11",
            freqs=[7.0,9.5,12.0],
            trial_sec=10.0,
            rest_sec=5.0,
            reps=1,
            outfile=f"recording{i}.csv",
            first_4_channels="--first-4-channels"
        )
        print(f"{'X'*90}")

        hzO = analyze_psd(
            df=data,
            freqs=[7.0,9.5,12.0],
            fmin=(7.0 - 6),
            fmax=(12.0 + 6),
            trial_sec=10.0,
            drop_sec=0.75,
            bp_low=(7.0 - 1.5),
            bp_high=(12.0 + 1.5),
            notch=60,
            per_target=False,
            welch_sec=2.0
        )
        if hzO != None:
            command += hzO

    print("COMMAND: ", command)
    match command:
      case "aa":
        output = "aa"
        print(output)
        return output
      case "ab":
        output = "ab"
        print(output)
        return output
      case "ac":
        output = "ac"
        print(output)
        return output
      case "ba":
        output = "ba"
        print(output)
        return output
      case "bb":
        output = "bb"
        print(output)
        return output
      case "bc":
        output = "bc"
        print(output)
        return output
      case "ca":
        output = "ca"
        print(output)
        return output
      case "cb":
        output = "cb"
        print(output)
        return output
      case "cc":
        output = "cc"
        print(output)
        return output
      case _:
        raise Exception("Unknown Command!")
    
ACTION_MAP = {
    "aa": "roll",
    "ab": "buy",
    "ac": "ok",
    "ba": "bid",
    "bb": "fold",
    "bc": "accept",
    "ca": "reject",
    "cb": "end_turn",
    "cc": "pay_jail",
}

def wait_for_eeg_action(expected_actions):
    """
    Call EEG experiment and return detected action.
    Retries if unexpected action is detected.
    """
    print(f"[DEBUG] Waiting for EEG action: {expected_actions}")
    
    while True:
        command = exp()
        print(f"[DEBUG] EEG returned command: {command}")
        
        action = ACTION_MAP.get(command)
        
        if action in expected_actions:
            print(f"[DEBUG] Got expected action: {action}")
            return action
        else:
            print(f"[DEBUG] Got unexpected action '{action}' from command '{command}', expected one of {expected_actions}")
            print("[DEBUG] Please try again...")
            time.sleep(1)
    
import pyautogui
import os
from PIL import Image
import pytesseract
import time

# 630 565 (BID)
# 630, 620 (FOLD)

# 1200 770 (START GAME)
# 555 425 (ROLL DICE)
# 410 460 (BUY), 350 620
# 540 420 (END TURN)

# 324 609 (pay 50 jail)

def move_and_click(x, y, delay=0.5):
  print(f"[DEBUG] Moving to ({x}, {y}) and clicking")
  pyautogui.moveTo(x, y)   # move to (x, y)
  time.sleep(0.5)
  pyautogui.click()
  time.sleep(delay)

def perform_ocr_on_region(x,y,x2,y2):
  w = x2 - x
  h = y2 - y
  region = (x, y, w, h)
  print(f"[DEBUG] Performing OCR on region: x={x}, y={y}, x2={x2}, y2={y2} (width={w}, height={h})")

  # Take a screenshot of that area
  img = pyautogui.screenshot(region=region)

  # Optional: improve text clarity
  img = img.convert("L")  # grayscale
  img = img.point(lambda x: 0 if x < 128 else 255)  # binarize

  # Run OCR
  text = pytesseract.image_to_string(img).strip().lower()
  print(f"[DEBUG] OCR result: '{text}'")
  return text


def bot_response():
  print("[DEBUG] Waiting for bot response (bid/fold or accept/reject)")
  
  while True:
    text = perform_ocr_on_region(228, 510, 707, 647)  # "fold" or "bid" / accept reject

    if "bid" in text and "fold" in text:
        print("[DEBUG] Bot response: BID")
        return "bid"
      
    elif "accept" in text and "reject" in text:
        print("[DEBUG] Bot response: ACCEPT")
        return "accept"
      
    prompt = player_prompt()
    if prompt == "roll" or prompt == "jail":
      state = "PLAYER"
      return "player"
      
    time.sleep(0.5)
    
  

def detect_first_player():
  print("[DEBUG] Detecting first player")
  text = perform_ocr_on_region(350, 260, 550, 300) # "SHOE STARTS FIRST"
  turn = False
  if "player" in text:
      turn = True
  print(f"[DEBUG] Player starts first: {turn}")
  time.sleep(1)
  move_and_click(530, 500 , delay=1)  # Click CLOSE
  return turn

def start_game():
  print("[DEBUG] ANY input to start")
  action = wait_for_eeg_action(ACTION_MAP.values())
  print("[DEBUG] Starting game")
  move_and_click(1200, 770, delay=2)  # Click START GAME

def focus_arc():
  print("[DEBUG] Focusing Arc browser")
  os.system('osascript -e \'tell application "Arc" to activate\'')
  
  
def roll_dice():
  print("[DEBUG] Waiting for 'l' key to roll dice")
  action = wait_for_eeg_action(['roll'])
  print("[DEBUG] Rolling dice")
  move_and_click(555, 425, delay=1)  # Click ROLL DICE
  
  print("[DEBUG] Checking for buy/ok prompt after roll")
  while True:
    text = perform_ocr_on_region(220, 580, 730, 650)
    
    text2 = perform_ocr_on_region(980, 442, 1300, 490)
    if "buy" in text:
      print("[DEBUG] Buy option detected")
      buy()
      return
    
    elif "ok" in text or "chance" in text2 or "community" in text2:
      print("[DEBUG] OK button detected")
      ok()
      return
    
    elif "jail" in text2:
      print("[DEBUG] HIT JAIL")
      end_turn() 
      return
    
    time.sleep(0.5)
   

def ok():
  print("[DEBUG] Waiting for 'l' key to click OK")
  action = wait_for_eeg_action(['ok'])
  print("[DEBUG] Clicking OK")
  move_and_click(475, 609, delay=1)  # Click OK

def buy():
  print("[DEBUG] Waiting for 'l' key to buy property")
  action = wait_for_eeg_action(['buy'])
  print("[DEBUG] Buying property")
  move_and_click(350, 620, delay=1)  # Click BUY
    
def jail(): # pay 20 or roll for doubles
  print("[DEBUG] In jail - waiting for 'l' key to pay")
  action = wait_for_eeg_action(['pay_jail'])
  print("[DEBUG] Paying jail fee")
  move_and_click(324, 609, delay=1)  # Click PAY 50

def player_prompt():
  # roll dice: 255 325 to 680 475
  text = perform_ocr_on_region(255, 325, 680, 675)
  if "roll" in text:
    print("[DEBUG] Player prompt: ROLL")
    return "roll"
  elif "pay" in text:   # need jail coord
    print("[DEBUG] Player prompt: JAIL")
    return "jail"
  print(f"[DEBUG] Player prompt: UNKNOWN (text: {text})")
  
  
def bid_fold():
  print("[DEBUG] Waiting for 'k' (BID) or 'l' (FOLD)")
  action = wait_for_eeg_action(['bid', 'fold'])

  if action == 'bid':
    print("[DEBUG] Player chose: BID")
    move_and_click(630, 565, delay=1)  # Click BID
  else:
    print("[DEBUG] Player chose: FOLD")
    move_and_click(630, 620, delay=1)  # Click FOLD
    
def accept_reject():
  print("[DEBUG] Waiting for 'k' (ACCEPT) or 'l' (REJECT)")
  action = wait_for_eeg_action(['accept', 'reject'])

  if action == 'accept':
    print("[DEBUG] Player chose: ACCEPT")
    move_and_click(630, 565, delay=1)  # Click ACCEPT
  else:
    print("[DEBUG] Player chose: REJECT")
    move_and_click(630, 620, delay=1)  # Click REJECT
  
def end_turn():
  print("[DEBUG] Waiting for 'l' key to end turn")
  action = wait_for_eeg_action(['end_turn'])
  print("[DEBUG] Ending turn")
  move_and_click(540, 420, delay=0.5)  # Click END TURN



if __name__ == "__main__":

  print("[DEBUG] Keyboard listener started")
  
  focus_arc()
  start_game()
  state = "INIT"
  
  turn = detect_first_player()
  state = "PLAYER" if turn else "BOT"
  print(f"[DEBUG] Initial state: {state}")


  while True:
    if state == "PLAYER":
      print(f"[DEBUG] === STATE: {state} ===")
      prompt = player_prompt()

      if prompt == "roll":
        roll_dice()
      elif prompt == "jail":
        jail()
        
      end_turn()

      state = "BOT"
      print(f"[DEBUG] Transitioning to state: {state}")
      
    elif state == "BOT":
      print(f"[DEBUG] === STATE: {state} ===")
      resp = bot_response() # bid, accept, or player
      
      if resp == "player":
        # Bot's turn ended, player needs to act
        print(f"[DEBUG] Bot turn ended, player's turn detected")
        state = "PLAYER"
        print(f"[DEBUG] Transitioning to state: {state}")
      elif resp == "bid":
        print(f"[DEBUG] Bot presented bid/fold choice")
        bid_fold()
        # After player responds, stay in BOT state to check what's next
      else:  # resp == "accept"
        print(f"[DEBUG] Bot presented accept/reject choice")
        accept_reject()
        # After player responds, stay in BOT state to check what's next