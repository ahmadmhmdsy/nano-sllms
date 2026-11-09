import os, sys
from huggingface_hub import snapshot_download
dest = r"E:\python projects\nano_SLMs\arabic_ocr\models\fixfips_khatt"
os.makedirs(dest, exist_ok=True)
p = snapshot_download(repo_id="FixFips/arabicocr-khatt", local_dir=dest)
print("OK", p)
for root, dirs, files in os.walk(dest):
    for f in files:
        fp = os.path.join(root, f)
        print(os.path.relpath(fp, dest), os.path.getsize(fp))
