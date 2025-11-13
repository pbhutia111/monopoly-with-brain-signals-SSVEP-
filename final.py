#!/usr/bin/env python3
"""
Combined BCI-controlled Monopoly bot using SSVEP (psy.py) + game automation (auto.py)
SSVEP output patterns control game actions instead of keyboard input.
"""

import pyautogui
import os
import time
import argparse
from typing import List, Optional, Dict
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft, fftfreq

# BrainFlow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# PsychoPy
from psychopy import visual, core, monitors


# =======================
# Configuration
# =======================

# Map SSVEP output patterns to game actions
ACTION_MAP = {
    # Single actions (for simple choices like OK, BUY, ROLL, etc.)
    "aaa": "CONFIRM",      # Default confirm action
    "bbb": "SKIP",         # Default skip/cancel action
    "ccc": "END_TURN",     # End turn explicitly
    
    # Binary choices (BID/FOLD, ACCEPT/REJECT)
    "abc": "BID",
    "bca": "FOLD",
    "cab": "ACCEPT", 
    "acb": "REJECT",
    
    # Property actions
    "aab": "BUY",
    "abb": "NO_BUY",
    
    # Jail actions
    "aac": "PAY_JAIL",
    "acc": "ROLL_JAIL",
}

# Game coordinates
COORDS = {
    'BID': (630, 565),
    'FOLD': (630, 620),
    'START_GAME': (1200, 770),
    'ROLL_DICE': (555, 425),
    'BUY': (350, 620),
    'OK': (475, 609),
    'END_TURN': (540, 420),
    'JAIL_PAY': (324, 609),
    'CLOSE': (530, 500),
    'ACCEPT': (630, 565),
    'REJECT': (630, 620)
}

# OCR regions
OCR_REGIONS = {
    'bot_response': (228, 510, 707, 647),
    'first_player': (350, 260, 550, 300),
    'player_prompt': (255, 325, 680, 675),
    'after_roll': (220, 580, 730, 650),
    'location': (980, 442, 1300, 490)
}


class SSVEPController:
    def __init__(self, board: str, serial_port: Optional[str], freqs: List[float] = [7.0, 9.5, 12.0]):
        self.board = board
        self.serial_port = serial_port
        self.freqs = freqs
        self.recorder = None
        self.win = None
        self.targets = None
        
    def setup(self):
        """Initialize display and BrainFlow recorder"""
        mon = monitors.Monitor('testMonitor')
        self.win = visual.Window(size=(1470, 800), fullscr=False, color='black', 
                                 units='norm', monitor=mon)
        
        self.targets = self._build_targets()
        self.recorder = self._setup_recorder()
        
    def _build_targets(self):
        """Build flicker targets in corners"""
        n = len(self.freqs)
        box_size = 0.25
        offset = 0.65
        
        corner_positions = [
            (-offset, offset),   # Top-Left
            (offset, offset),    # Top-Right
            (-offset, -offset),  # Bottom-Left
            (offset, -offset)    # Bottom-Right
        ]
        
        if n == 2:
            positions = [corner_positions[0], corner_positions[3]]
        elif n == 3:
            positions = [corner_positions[0], corner_positions[1], corner_positions[2]]
        else:
            positions = corner_positions[:n]
        
        targets = []
        corner_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        
        for i, f in enumerate(self.freqs):
            pos = positions[i]
            rect = visual.Rect(self.win, width=box_size, height=box_size,
                             fillColor="white", lineColor="white",
                             pos=pos, opacity=0.0)
            
            label_offset = 0.15
            lbl = visual.TextStim(self.win, 
                                text=f"{f:.1f} Hz\n{corner_labels[i]}",
                                pos=(pos[0], pos[1] - label_offset), 
                                height=0.05, color='white')
            lbl.draw()
            
            targets.append({'freq': f, 'rect': rect, 'label': corner_labels[i]})
        
        fixation = visual.TextStim(self.win, text='+', height=0.1, 
                                  color='gray', pos=(0, 0))
        fixation.draw()
        self.win.flip()
        
        return targets
    
    def _setup_recorder(self):
        """Setup BrainFlow board"""
        params = BrainFlowInputParams()
        
        if self.board.lower() == "cyton":
            params.serial_port = self.serial_port
            board_id = BoardIds.CYTON_BOARD.value
        elif self.board.lower() == "ganglion":
            params.serial_port = self.serial_port
            board_id = BoardIds.GANGLION_BOARD.value
        elif self.board.lower() == "synthetic":
            board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            raise ValueError("Unknown board type")
        
        board = BoardShim(board_id, params)
        sr = BoardShim.get_sampling_rate(board_id)
        
        board.prepare_session()
        board.start_stream(num_samples=int(300 * sr))
        
        return {'board': board, 'board_id': board_id, 'sr': sr}
    
    def run_trial(self, trial_sec: float = 10.0, prompt: str = "Focus on targets") -> str:
        """Run one SSVEP trial and return action code (e.g., 'a', 'b', 'c')"""
        print(f"[SSVEP] Running trial: {prompt}")
        
        # Show prompt
        info = visual.TextStim(self.win, text=prompt, pos=(0, 0.6), 
                              height=0.05, color='white')
        
        # Run flicker
        clock = core.Clock()
        clock.reset()
        
        while clock.getTime() < trial_sec:
            t = clock.getTime()
            info.draw()
            for target in self.targets:
                freq = target['freq']
                frac = (freq * t) % 1.0
                target['rect'].opacity = 1.0 if frac < 0.5 else 0.0
                target['rect'].draw()
            self.win.flip()
        
        # Get data and analyze
        board = self.recorder['board']
        data = board.get_current_board_data(int(trial_sec * self.recorder['sr']))
        
        # Analyze and return action
        action_code = self._analyze_data(data)
        print(f"[SSVEP] Trial result: {action_code}")
        
        return action_code
    
    def _analyze_data(self, data: np.ndarray) -> str:
        """Analyze EEG data and return action code"""
        board_id = self.recorder['board_id']
        eeg_channels = BoardShim.get_eeg_channels(board_id)[:4]
        sr = self.recorder['sr']
        
        eeg_data = data[eeg_channels, :].T  # Shape: (n_samples, 4)
        
        # Bandpass filter
        eeg_filtered = self._bandpass_filter(eeg_data, 5.5, 15, sr)
        eeg_filtered = self._notch_filter(eeg_filtered, 60, sr)
        
        # Windowing
        windows = []
        window_size = 200
        step_size = 50
        
        for start in range(0, len(eeg_filtered) - window_size, step_size):
            end = start + window_size
            windows.append(eeg_filtered[start:end, :])
        
        if not windows:
            return 'b'  # Default fallback
        
        # FFT analysis
        target_freqs = [7.0, 9.5, 12.0]
        freq_powers = {freq: [] for freq in target_freqs}
        
        for window in windows:
            for ch in range(window.shape[1]):
                channel_data = window[:, ch]
                fft_values = fft(channel_data)
                fft_freqs = fftfreq(len(channel_data), 1/sr)
                power_spectrum = np.abs(fft_values[:len(channel_data)//2])
                positive_freqs = fft_freqs[:len(channel_data)//2]
                
                for target_freq in target_freqs:
                    idx = np.argmin(np.abs(positive_freqs - target_freq))
                    freq_powers[target_freq].append(power_spectrum[idx])
        
        # Calculate average powers
        avg_powers = {freq: np.mean(powers) for freq, powers in freq_powers.items()}
        
        power_7 = avg_powers[7.0]
        power_9 = avg_powers[9.5]
        power_12 = avg_powers[12.0]
        
        print(f"[SSVEP] Powers - 7Hz: {power_7:.2f}, 9.5Hz: {power_9:.2f}, 12Hz: {power_12:.2f}")
        
        # Determine which frequency has highest power
        if power_7 > power_9 and power_7 > power_12:
            return 'a'
        elif power_9 > power_7 and power_9 > power_12:
            return 'b'
        else:
            return 'c'
    
    def _bandpass_filter(self, data, lowcut, highcut, fs):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    def _notch_filter(self, data, freq, fs):
        nyquist = fs / 2
        b, a = iirnotch(freq / nyquist, 30)
        return filtfilt(b, a, data, axis=0)
    
    def cleanup(self):
        """Stop recording and close window"""
        if self.recorder:
            self.recorder['board'].stop_stream()
            self.recorder['board'].release_session()
        if self.win:
            self.win.close()


# =======================
# Game Automation (from auto.py)
# =======================

class MonopolyBot:
    def __init__(self, ssvep: SSVEPController):
        self.ssvep = ssvep
        
    def move_and_click(self, x, y, delay=0.5):
        print(f"[GAME] Clicking ({x}, {y})")
        pyautogui.moveTo(x, y)
        time.sleep(0.5)
        pyautogui.click()
        time.sleep(delay)
    
    def perform_ocr_on_region(self, x, y, x2, y2):
        w = x2 - x
        h = y2 - y
        img = pyautogui.screenshot(region=(x, y, w, h))
        img = img.convert("L")
        img = img.point(lambda x: 0 if x < 128 else 255)
        text = pytesseract.image_to_string(img).strip().lower()
        print(f"[OCR] Region ({x},{y},{x2},{y2}): '{text}'")
        return text
    
    def get_bci_action(self, prompt: str, num_trials: int = 3) -> str:
        """Run multiple SSVEP trials and return consensus action pattern"""
        command = ""
        for i in range(num_trials):
            result = self.ssvep.run_trial(trial_sec=10.0, prompt=prompt)
            command += result
            core.wait(1.0)
        
        print(f"[BCI] Command sequence: {command}")
        return command
    
    def execute_action(self, action: str):
        """Execute a mapped game action"""
        if action in COORDS:
            x, y = COORDS[action]
            self.move_and_click(x, y, delay=1)
        else:
            print(f"[GAME] Unknown action: {action}")
    
    def focus_arc(self):
        print("[GAME] Focusing Arc browser")
        os.system('osascript -e \'tell application "Arc" to activate\'')
    
    def start_game(self):
        print("[GAME] Starting game...")
        command = self.get_bci_action("Focus to START GAME", num_trials=1)
        self.move_and_click(*COORDS['START_GAME'], delay=2)
    
    def detect_first_player(self):
        print("[GAME] Detecting first player")
        text = self.perform_ocr_on_region(*OCR_REGIONS['first_player'])
        turn = "player" in text
        print(f"[GAME] Player starts first: {turn}")
        time.sleep(1)
        self.move_and_click(*COORDS['CLOSE'], delay=1)
        return turn
    
    def roll_dice(self):
        print("[GAME] Rolling dice...")
        command = self.get_bci_action("Focus to ROLL DICE", num_trials=1)
        self.move_and_click(*COORDS['ROLL_DICE'], delay=1)
        
        # Check what happened after roll
        while True:
            text = self.perform_ocr_on_region(*OCR_REGIONS['after_roll'])
            text2 = self.perform_ocr_on_region(*OCR_REGIONS['location'])
            
            if "buy" in text:
                print("[GAME] Buy option detected")
                self.buy_property()
                return
            elif "ok" in text or "chance" in text2 or "community" in text2:
                print("[GAME] OK button detected")
                self.click_ok()
                return
            elif "jail" in text2:
                print("[GAME] Hit jail")
                self.end_turn()
                return
            
            time.sleep(0.5)
    
    def buy_property(self):
        print("[GAME] Deciding whether to buy...")
        command = self.get_bci_action("Focus: BUY property?", num_trials=3)
        
        if command in ACTION_MAP:
            action = ACTION_MAP[command]
            if action == "BUY":
                self.move_and_click(*COORDS['BUY'], delay=1)
            else:
                print("[GAME] Skipping purchase")
        else:
            # Default to buying
            self.move_and_click(*COORDS['BUY'], delay=1)
    
    def click_ok(self):
        print("[GAME] Clicking OK...")
        command = self.get_bci_action("Focus to click OK", num_trials=1)
        self.move_and_click(*COORDS['OK'], delay=1)
    
    def jail(self):
        print("[GAME] In jail, deciding action...")
        command = self.get_bci_action("Focus: PAY jail fee?", num_trials=3)
        
        if command in ACTION_MAP:
            action = ACTION_MAP[command]
            if action == "PAY_JAIL":
                self.move_and_click(*COORDS['JAIL_PAY'], delay=1)
        else:
            # Default to paying
            self.move_and_click(*COORDS['JAIL_PAY'], delay=1)
    
    def bid_fold(self):
        print("[GAME] Deciding BID or FOLD...")
        command = self.get_bci_action("Focus: BID or FOLD?", num_trials=3)
        
        if command in ACTION_MAP:
            action = ACTION_MAP[command]
            if action == "BID":
                self.move_and_click(*COORDS['BID'], delay=1)
            else:
                self.move_and_click(*COORDS['FOLD'], delay=1)
        else:
            # Default to FOLD
            self.move_and_click(*COORDS['FOLD'], delay=1)
    
    def accept_reject(self):
        print("[GAME] Deciding ACCEPT or REJECT...")
        command = self.get_bci_action("Focus: ACCEPT or REJECT?", num_trials=3)
        
        if command in ACTION_MAP:
            action = ACTION_MAP[command]
            if action == "ACCEPT":
                self.move_and_click(*COORDS['ACCEPT'], delay=1)
            else:
                self.move_and_click(*COORDS['REJECT'], delay=1)
        else:
            # Default to REJECT
            self.move_and_click(*COORDS['REJECT'], delay=1)
    
    def end_turn(self):
        print("[GAME] Ending turn...")
        command = self.get_bci_action("Focus to END TURN", num_trials=1)
        self.move_and_click(*COORDS['END_TURN'], delay=0.5)
    
    def player_prompt(self):
        text = self.perform_ocr_on_region(*OCR_REGIONS['player_prompt'])
        if "roll" in text:
            print("[GAME] Player prompt: ROLL")
            return "roll"
        elif "pay" in text:
            print("[GAME] Player prompt: JAIL")
            return "jail"
        print(f"[GAME] Player prompt: UNKNOWN")
        return None
    
    def bot_response(self):
        print("[GAME] Waiting for bot response...")
        
        while True:
            text = self.perform_ocr_on_region(*OCR_REGIONS['bot_response'])
            
            if "bid" in text and "fold" in text:
                print("[GAME] Bot response: BID/FOLD choice")
                return "bid"
            elif "accept" in text and "reject" in text:
                print("[GAME] Bot response: ACCEPT/REJECT choice")
                return "accept"
            
            prompt = self.player_prompt()
            if prompt in ["roll", "jail"]:
                return "player"
            
            time.sleep(0.5)
    
    def run_game(self):
        """Main game loop"""
        self.focus_arc()
        self.start_game()
        
        turn = self.detect_first_player()
        state = "PLAYER" if turn else "BOT"
        print(f"[GAME] Initial state: {state}")
        
        while True:
            try:
                if state == "PLAYER":
                    print(f"[GAME] === STATE: {state} ===")
                    prompt = self.player_prompt()
                    
                    if prompt == "roll":
                        self.roll_dice()
                    elif prompt == "jail":
                        self.jail()
                    
                    self.end_turn()
                    state = "BOT"
                    print(f"[GAME] Transitioning to: {state}")
                
                elif state == "BOT":
                    print(f"[GAME] === STATE: {state} ===")
                    resp = self.bot_response()
                    
                    if resp == "player":
                        print("[GAME] Bot turn ended")
                        state = "PLAYER"
                    elif resp == "bid":
                        self.bid_fold()
                    elif resp == "accept":
                        self.accept_reject()
                    
            except KeyboardInterrupt:
                print("[GAME] Stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                raise


# =======================
# Main
# =======================

def main():
    parser = argparse.ArgumentParser(description="BCI-controlled Monopoly bot")
    parser.add_argument("--board", type=str, choices=["cyton", "synthetic", "ganglion"], 
                       default="synthetic", help="BrainFlow board type")
    parser.add_argument("--serial", type=str, default=None,
                       help="Serial port for Cyton/Ganglion")
    parser.add_argument("--freqs", type=float, nargs="+", default=[7.0, 9.5, 12.0],
                       help="SSVEP frequencies")
    
    args = parser.parse_args()
    
    print("[MAIN] Initializing BCI-controlled Monopoly bot...")
    
    # Setup SSVEP controller
    ssvep = SSVEPController(board=args.board, serial_port=args.serial, freqs=args.freqs)
    ssvep.setup()
    
    # Create and run game bot
    bot = MonopolyBot(ssvep)
    
    try:
        bot.run_game()
    finally:
        ssvep.cleanup()
        print("[MAIN] Cleanup complete")


if __name__ == "__main__":
    main()
