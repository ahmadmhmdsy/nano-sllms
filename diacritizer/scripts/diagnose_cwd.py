import os, sys
f = os.path.abspath(__file__)
print("f:", f)
print("dir1:", os.path.dirname(f))
print("dir2:", os.path.dirname(os.path.dirname(f)))
print("model_dir_expected:", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(f))), "models", "e19", "tashkeel-350m-v2"))
print("cwd before:", os.getcwd())
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(f))) if __name__ else os.getcwd())
print("exists smoke:", os.path.exists("models/e19/inputs/smoke.bare.txt"))