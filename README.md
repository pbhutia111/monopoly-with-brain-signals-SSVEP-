# 🎮 SSVEP BCI Pong Game

A brain-computer interface (BCI) implementation of the classic Pong game using Steady-State Visual Evoked Potential (SSVEP) signals. This project combines neuroscience with gaming to create an innovative hands-free control system.

## 🧠 Overview

This project implements a Pong game that can be controlled using brain signals (SSVEP) detected through an EEG system. The game features flickering stimuli at different frequencies that evoke measurable brain responses, which can be used to control the game paddles.

## ✨ Features

- **Dual Control Modes**: Traditional keyboard controls (A/D and Arrow keys) and BCI-based control
- **SSVEP Stimulation**: Flickering visual stimuli at 10Hz and 15Hz frequencies
- **Real-time Gameplay**: Smooth 60 FPS gaming experience
- **Color-coded Paddles**: Orange (top) and Pink (bottom) for easy distinction
- **Dynamic Ball Physics**: Increasing speed with each paddle hit
- **Score Tracking**: Keep track of points for both players

## 🛠️ Technologies Used

- **PsychoPy**: Visual stimulus presentation and window management
- **NumPy**: Numerical computations
- **Pyglet**: Low-level keyboard input handling

## 📋 Prerequisites

- Python 3.7+
- EEG hardware (for BCI control) - Optional for keyboard-only mode

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/ssvep-bci-pong.git
cd ssvep-bci-pong
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

## 🎯 Usage

### Keyboard Mode

Run the game with standard keyboard controls:

```bash
python pong.py
```

**Controls:**
- **Top Paddle**: `A` (left) / `D` (right)
- **Bottom Paddle**: `←` (left) / `→` (right)
- **Exit**: `ESC`

### BCI Mode

For brain-computer interface control, you'll need:
1. An EEG headset compatible with OpenBCI or similar
2. Additional signal processing code (see other files in the project)
3. Calibration of SSVEP frequencies (10Hz left, 15Hz right)

## 📁 Project Structure

```
.
├── pong.py              # Main game implementation
├── ssvep_experiment.py  # SSVEP BCI experiment with CCA classification
├── auto.py              # Automated control scripts
├── final.py             # Integrated BCI-game system
├── mouse_controls.py    # Mouse-based control alternative
├── psy.py               # PsychoPy utilities
├── analyze.ipynb        # Data analysis notebook
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎨 Game Mechanics

### Ball Physics
- Initial speed: 4 pixels/frame
- Speed increase: +0.5 per paddle hit
- Maximum speed: 10 pixels/frame
- Direction: Reverses on paddle collision

### Paddle Specifications
- Width: 300 pixels
- Height: 15 pixels
- Speed: 10 pixels/frame
- Boundary collision detection

### SSVEP Stimuli
- Left flicker box: 10Hz frequency
- Right flicker box: 15Hz frequency
- Size: 50x50 pixels
- Position: Adjacent to bottom paddle

## 🔬 How SSVEP Works

SSVEP (Steady-State Visual Evoked Potential) is a natural response that occurs in your brain when you look at a flickering light. When you focus on a stimulus flickering at a specific frequency (e.g., 10Hz), your brain produces electrical activity at that same frequency. This can be detected using EEG and used to determine what you're looking at.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classic Pong and modern BCI research
- Built with PsychoPy for precise visual stimulus timing
- SSVEP methodology based on brain-computer interface research

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is a research/educational project. For actual BCI implementation, proper calibration and signal processing are required.
