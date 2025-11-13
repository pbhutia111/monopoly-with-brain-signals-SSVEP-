import pyautogui
import os
from PIL import Image
import pytesseract
from pynput import keyboard
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
  pyautogui.moveTo(x, y)
  time.sleep(0.5)
  pyautogui.click()
  time.sleep(delay)

def perform_ocr_on_region(x,y,x2,y2):
  w = x2 - x
  h = y2 - y
  region = (x, y, w, h)
  print(f"[DEBUG] Performing OCR on region: x={x}, y={y}, x2={x2}, y2={y2} (width={w}, height={h})")

  img = pyautogui.screenshot(region=region)

  img = img.convert("L")
  img = img.point(lambda x: 0 if x < 128 else 255)  # binarize

  text = pytesseract.image_to_string(img).strip().lower()
  print(f"[DEBUG] OCR result: '{text}'")
  return text
  
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
  move_and_click(1200, 770, delay=1)  # Click START GAME

def roll_dice():
  move_and_click(555, 425, delay=1)  # Click ROLL DICE
   
def ok():
  move_and_click(475, 609, delay=1)  # Click OK

def buy():
  move_and_click(350, 620, delay=1)  # Click BUY
    
def pay_jail(): # pay 20 or roll for doubles
  move_and_click(324, 609, delay=1)  # Click PAY 50
    
def bid():
  move_and_click(630, 565, delay=1)  # Click BID
  
def fold():
  move_and_click(630, 620, delay=1)  # Click FOLD
    
def reject():
  move_and_click(630, 620, delay=1)  # Click REJECT
  
def accept():
  move_and_click(630, 565, delay=1)  # Click ACCEPT
  
def end_turn():
  move_and_click(540, 420, delay=0.5)  # Click END TURN
  


