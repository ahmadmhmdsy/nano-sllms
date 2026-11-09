import sys
sys.path.append(r"E:\python projects\nano_SLMs\.venv-paddle\Lib\site-packages")
try:
    import cv2; print("BORROW-paddle OK", cv2.__version__)
except Exception as e:
    print("BORROW-paddle FAIL", type(e).__name__, e)

sys.path.append(r"E:\python projects\nano_SLMs\.venv-paddle-gpu\Lib\site-packages")
try:
    import cv2; print("BORROW-paddle-gpu OK", cv2.__version__)
except Exception as e:
    print("BORROW-paddle-gpu FAIL", type(e).__name__, e)
