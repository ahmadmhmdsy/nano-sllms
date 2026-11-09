import importlib.util, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
p = Path(r"E:\python projects\nano_SLMs\arabic_ocr\models\fixfips_khatt\author_code\preprocess.py")
spec = importlib.util.spec_from_file_location("prep", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
img = Image.open(r"E:\python projects\nano_SLMs\data\hflines\val\000000.png")
print("orig mode/size:", img.mode, img.size)
g = m.to_grayscale(img)
b = m.binarize(g)
print("binarized:", b.mode, b.size, "blackfrac=%.3f" % ((np.array(b) == 0).mean()))
r = m.resize_keep_ratio_height(b, 96)
print("resized:", r.size)
if r.width > 1536: r = r.resize((1536, 96), Image.LANCZOS)
pad = m.pad_width(r, 1536)
print("padded:", pad.size)
arr = np.asarray(pad, dtype=np.float32)/255.0
print("tensor:", arr.shape, "min", arr.min(), "max", arr.max())
