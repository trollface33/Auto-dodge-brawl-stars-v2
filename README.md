"""
README for Auto-dodge-brawl-stars-v2.
"""

# Auto-dodge-brawl-stars-v2

This version is a cleaner detection pipeline based on:
- ADB screen capture
- object tracking across several frames
- motion + distance filtering
- threat scoring
- dodge direction planning
- dry-run mode before real input

## Quick start

1. Install dependencies:
   python -m pip install -r requirements.txt

2. Make sure your phone is connected and authorized with ADB:
   adb devices

3. Edit config.json and set your device serial.

4. Run the bot:
   python main.py --device 3289SH1010010782

5. Dry-run only:
   python main.py --dry-run --device 3289SH1010010782

## Notes

This project is intentionally conservative and can still trigger false positives depending on the game scene and UI colors. The dry-run mode is the best way to validate detection before enabling real swipes.
