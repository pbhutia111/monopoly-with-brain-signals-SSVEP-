# 🧠� Brain-Controlled Monopoly Bot

Play Monopoly using only your **brain signals**! This project uses SSVEP (Steady-State Visual Evoked Potential) brain-computer interface technology to control an automated Monopoly bot through EEG signals, allowing completely hands-free gameplay.

## 🧠 Overview

This groundbreaking system combines brain-computer interface (BCI) technology with game automation to play Monopoly using nothing but your thoughts. By focusing on flickering visual stimuli at different frequencies (7Hz, 9.5Hz, and 12Hz), users generate detectable brain signals that are classified using Canonical Correlation Analysis (CCA) to issue game commands.

The system records EEG data via BrainFlow (supporting OpenBCI Cyton/Ganglion boards), analyzes SSVEP responses in real-time, and translates them into Monopoly actions like rolling dice, buying properties, bidding on auctions, and making strategic decisions—all without touching a keyboard or mouse!

## ✨ Key Features

- **🧠 Hands-Free BCI Control**: Play Monopoly using only brain signals via EEG
- **⚡ Real-Time SSVEP Classification**: CCA-based frequency detection (7Hz, 9.5Hz, 12Hz)
- **🤖 Intelligent Game Automation**: OCR-based game state detection and automated clicking
- **🎯 Multi-Action Command System**: 9 distinct commands from 3-frequency combinations
- **🔬 Full Signal Processing Pipeline**: Bandpass filtering, notch filtering, FFT analysis
- **🎮 Complete Game Integration**: Handles all Monopoly actions (buy, bid, fold, jail, etc.)
- **📊 Data Recording & Analysis**: Save EEG data for offline analysis

## 🛠️ Technologies Used

- **BrainFlow**: Real-time EEG data acquisition from OpenBCI hardware
- **PsychoPy**: High-precision visual stimulus presentation for SSVEP
- **scikit-learn**: Canonical Correlation Analysis (CCA) for signal classification
- **SciPy**: Signal processing (bandpass, notch filters, FFT, Welch PSD)
- **NumPy & Pandas**: Numerical computation and data handling
- **PyAutoGUI**: Game automation and mouse control
- **Tesseract OCR**: Game state detection via screen reading

## 📋 Prerequisites

- **Python 3.7+**
- **EEG Hardware**: OpenBCI Cyton or Ganglion board (or use synthetic mode for testing)
- **macOS** (Arc browser automation - adaptable for other OS/browsers)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/pbhutia111/monopoly-with-brain-signals-SSVEP-.git
cd monopoly-with-brain-signals-SSVEP-
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows - download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

## 🎯 Usage

### Option 1: Full BCI Control (Requires EEG Hardware)

Run the complete brain-controlled Monopoly bot:

```bash
# With OpenBCI Ganglion
python psy.py

# With OpenBCI Cyton
python ssvep_experiment.py run --board cyton --serial /dev/tty.usbserial-XXXX --freqs 7 9.5 12 --trial-sec 10 --reps 2

# Testing with synthetic data (no hardware)
python ssvep_experiment.py run --board synthetic --freqs 7 9.5 12 --trial-sec 10 --reps 2
```

**How it works:**
1. Focus on Arc browser with Monopoly game open
2. Look at flickering boxes to generate SSVEP signals
3. System detects your brain signals and executes game actions
4. Commands are formed from 2 trials × 3 frequencies = 9 possible actions

### Option 2: Keyboard Control (No EEG Required)

Test the automation with keyboard controls:

```bash
python auto.py
```

**Controls:**
- **`L` key**: Confirm actions (roll, buy, OK, end turn)
- **`K` key**: Accept/Bid
- **`L` key**: Reject/Fold

### Option 3: Pong SSVEP Demo

Test SSVEP stimulus with a simple Pong game:

```bash
python pong.py
```

**Controls:**
- **Top Paddle**: `A` (left) / `D` (right)
- **Bottom Paddle**: `←` (left) / `→` (right)
- **Exit**: `ESC`

## 📁 Project Structure

```
.
├── psy.py               # Main BCI-controlled Monopoly bot (full integration)
├── ssvep_experiment.py  # SSVEP experiment runner with CCA classification
├── auto.py              # Monopoly automation with keyboard control
├── final.py             # Alternative BCI+game integration (experimental)
├── mouse_controls.py    # Mouse automation utilities
├── pong.py              # SSVEP Pong demo/test game
├── analyze.ipynb        # EEG data analysis notebook
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🧬 How It Works

### 1. SSVEP Signal Generation
The system displays flickering boxes at three distinct frequencies:
- **7.0 Hz** → Command "a"
- **9.5 Hz** → Command "b"  
- **12.0 Hz** → Command "c"

### 2. Brain Signal Detection
When you focus on a flickering box, your visual cortex generates electrical activity at the same frequency. This SSVEP response is recorded via EEG electrodes (4-channel OpenBCI setup).

### 3. Signal Processing Pipeline
```
Raw EEG → Bandpass Filter (5.5-15Hz) → Notch Filter (60Hz) → 
Sliding Windows (1s) → FFT Analysis → Power Spectrum Extraction
```

### 4. CCA Classification
Canonical Correlation Analysis (CCA) compares recorded signals with reference sine/cosine waves at target frequencies to determine which frequency you're focusing on.

### 5. Command Mapping
Two consecutive trials create a 2-character command (e.g., "ab", "ca", "bb"):

| Command | Action | Usage |
|---------|--------|-------|
| `aa` | Start Game | Begin new game |
| `ab` | Roll Dice | Roll on your turn |
| `ac` | OK/Confirm | Acknowledge cards/events |
| `ba` | Buy Property | Purchase property |
| `bb` | Pay Jail | Pay $50 to exit jail |
| `bc` | Bid | Place bid in auction |
| `ca` | Fold | Skip auction |
| `cb` | Reject | Decline trade offer |
| `cc` | Accept | Accept trade offer |

### 6. Game Automation
- **OCR Detection**: Reads game state from screen
- **State Machine**: Tracks player/bot turns
- **PyAutoGUI**: Executes mouse clicks at detected UI elements

## 🔬 Technical Details

### SSVEP Stimulus Parameters
- Display size: 1470×800 pixels
- Flicker box size: 25% of screen
- Corner positions: Top-Left (7Hz), Top-Right (9.5Hz), Bottom-Left (12Hz)
- Duty cycle: 50%
- Trial duration: 10 seconds
- Rest period: 4-5 seconds

### Signal Processing
- **Sampling rate**: 200 Hz (Ganglion)
- **Bandpass**: 5.5-15 Hz (4th order Butterworth)
- **Notch**: 60 Hz (line noise removal)
- **Window size**: 200 samples (1 second)
- **Step size**: 50 samples (250ms overlap)
- **FFT resolution**: 0.5 Hz bins from 3-20 Hz

### Classification
- **Method**: Canonical Correlation Analysis (CCA)
- **Harmonics**: 2 (fundamental + 2nd harmonic)
- **Reference signals**: Sine/cosine pairs for each frequency
- **Decision**: Majority vote across sliding windows

## 🎮 Game State Machine

The bot operates as a finite state machine:

```
START → PLAYER TURN → [Roll → Buy/OK/Jail] → End Turn → BOT TURN → [Wait/Bid/Accept] → PLAYER TURN
```

**Player State:**
- Detect prompt (roll dice or jail)
- Execute action via BCI
- Handle post-roll events (buy, chance/community chest)
- End turn

**Bot State:**  
- Monitor for bid/fold or accept/reject prompts
- Respond via BCI when player input required
- Transition to player turn when bot finishes

## 🐛 Troubleshooting

**EEG signals not detected:**
- Check electrode impedance (should be <40kΩ)
- Ensure good scalp contact with conductive gel
- Verify BrainFlow is receiving data: check console output

**OCR not working:**
- Ensure Tesseract is installed and in PATH
- Verify game window is visible and unobstructed
- Adjust OCR region coordinates if using different screen resolution

**Flicker boxes not visible:**
- Check PsychoPy window is focused
- Verify monitor refresh rate supports frequencies (≥60Hz recommended)
- Try adjusting window size or position

**Commands not executing:**
- Verify Arc browser is running and game is loaded
- Check PyAutoGUI screen coordinates match your display
- Enable debug logging to see command detection

## 📊 Data Analysis

The `analyze.ipynb` notebook provides tools for:
- Visualizing raw EEG signals
- Computing power spectral density (PSD)
- Analyzing SSVEP response strength
- Comparing classification accuracy
- Debugging signal quality issues

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for additional EEG hardware (Emotiv, NeuroSky, etc.)
- Improved classification algorithms (xDAWN, TRCA, DNN)
- Adaptive frequency selection
- Multi-player support
- Cross-platform browser automation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenBCI for open-source EEG hardware and BrainFlow library
- PsychoPy team for precise stimulus presentation
- SSVEP-BCI research community for methodology
- Monopoly game mechanics (Hasbro)

## 📧 Contact

For questions, collaboration, or bug reports, please open an issue on GitHub.

## 🎓 Research Context

This project was developed as a brain-computer interface experiment demonstrating:
1. Real-time SSVEP classification for command & control
2. Integration of BCI with complex game automation
3. Practical application of signal processing techniques
4. Hands-free human-computer interaction

**Note**: This is a research/educational project. Proper EEG electrode placement, calibration, and signal quality are essential for reliable BCI control. Results may vary based on individual neurophysiology and environmental factors.

---

**Built with 🧠 + ⚡ by [pbhutia111](https://github.com/pbhutia111)**
