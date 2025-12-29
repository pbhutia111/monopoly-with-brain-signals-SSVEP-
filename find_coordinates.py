#!/usr/bin/env python3
"""
Helper script to find button coordinates for Monopoly game ONE AT A TIME.
Run this script multiple times, once for each button.
"""

import pyautogui
import time
import sys

print("="*60)
print("MONOPOLY COORDINATE FINDER - ONE BUTTON AT A TIME")
print("="*60)

buttons = {
    "1": "START GAME (NEW GAME button)",
    "2": "ROLL DICE",
    "3": "BUY property",
    "4": "OK (after rolling/landing)",
    "5": "PAY JAIL (pay $50 to get out)",
    "6": "BID (in auction)",
    "7": "FOLD (in auction)",
    "8": "REJECT (trade offer)",
    "9": "ACCEPT (trade offer)",
    "10": "CLOSE (any popup/dialog)"
}

print("\nWhich button do you want to find?")
print("-" * 60)
for key, desc in buttons.items():
    print(f"  {key:2}. {desc}")
print("-" * 60)

choice = input("\nEnter number (or 'q' to quit): ").strip()

if choice.lower() == 'q':
    print("Exiting...")
    sys.exit(0)

if choice not in buttons:
    print(f"Invalid choice: {choice}")
    sys.exit(1)

button_name = buttons[choice]

print(f"\n📍 Finding coordinates for: {button_name}")
print("\nInstructions:")
print("  1. Make sure Monopoly game is open and visible")
print("  2. Move your mouse OVER the button")
print("  3. Keep your mouse STILL")
print("  4. Countdown will start in 3 seconds...")

time.sleep(3)

print("\n⏱️  Capturing in...")
for i in range(3, 0, -1):
    print(f"   {i}...")
    time.sleep(1)

x, y = pyautogui.position()

print(f"\n✅ CAPTURED: ({x}, {y})")
print("\n" + "="*60)
print(f"Button: {button_name}")
print(f"Coordinates: x={x}, y={y}")
print("="*60)

print("\n📋 Copy this line for your notes:")
print(f"   {choice}. {button_name} → ({x}, {y})")

print("\n💡 Run this script again to capture another button!")
print("   Or type 'q' to quit.")
