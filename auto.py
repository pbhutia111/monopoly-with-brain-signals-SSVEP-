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

# Global variables to track key states
l_action = False
k_action = False

def on_press(key):
    global l_action, k_action
    try:
        if key.char == 'l':
            l_action = True
            print("[DEBUG] Key 'l' pressed - action flagged")
        elif key.char == 'k':
            k_action = True
            print("[DEBUG] Key 'k' pressed - action flagged")
    except AttributeError:
        pass

def on_release(key):
    # No longer resetting flags on release
    pass

def wait_for_l():
    """Wait for 'l' key press and consume the action flag"""
    global l_action
    print("[DEBUG] Waiting for 'l' key...")
    while not l_action:
        time.sleep(0.1)
    l_action = False  # Consume the action
    print("[DEBUG] 'l' key action consumed")

def wait_for_k_or_l():
    """Wait for 'k' or 'l' key press and return which was pressed"""
    global k_action, l_action
    print("[DEBUG] Waiting for 'k' or 'l' key...")
    while not k_action and not l_action:
        time.sleep(0.1)
    
    if k_action:
        k_action = False  # Consume the action
        print("[DEBUG] 'k' key action consumed")
        return 'k'
    else:
        l_action = False  # Consume the action
        print("[DEBUG] 'l' key action consumed")
        return 'l'

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
  print("[DEBUG] Waiting for 'l' key to start game")
  wait_for_l()
  print("[DEBUG] Starting game")
  move_and_click(1200, 770, delay=2)  # Click START GAME

def focus_arc():
  print("[DEBUG] Focusing Arc browser")
  os.system('osascript -e \'tell application "Arc" to activate\'')
  
  
def roll_dice():
  print("[DEBUG] Waiting for 'l' key to roll dice")
  wait_for_l()
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
  wait_for_l()
  print("[DEBUG] Clicking OK")
  move_and_click(475, 609, delay=1)  # Click OK

def buy():
  print("[DEBUG] Waiting for 'l' key to buy property")
  wait_for_l()
  print("[DEBUG] Buying property")
  move_and_click(350, 620, delay=1)  # Click BUY
    
def jail(): # pay 20 or roll for doubles
  print("[DEBUG] In jail - waiting for 'l' key to pay")
  wait_for_l()
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
  key = wait_for_k_or_l()
  
  if key == 'k':
    print("[DEBUG] Player chose: BID")
    move_and_click(630, 565, delay=1)  # Click BID
  else:
    print("[DEBUG] Player chose: FOLD")
    move_and_click(630, 620, delay=1)  # Click FOLD
    
def accept_reject():
  print("[DEBUG] Waiting for 'k' (ACCEPT) or 'l' (REJECT)")
  key = wait_for_k_or_l()
  
  if key == 'k':
    print("[DEBUG] Player chose: ACCEPT")
    move_and_click(630, 565, delay=1)  # Click ACCEPT
  else:
    print("[DEBUG] Player chose: REJECT")
    move_and_click(630, 620, delay=1)  # Click REJECT
  
def end_turn():
  print("[DEBUG] Waiting for 'l' key to end turn")
  wait_for_l()
  print("[DEBUG] Ending turn")
  move_and_click(540, 420, delay=0.5)  # Click END TURN
  
if __name__ == "__main__":
  print("[DEBUG] Starting bot...")
  # Start keyboard listener
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()
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






