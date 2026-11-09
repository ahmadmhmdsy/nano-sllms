import torch, json
p = r"E:\python projects\nano_SLMs\arabic_ocr\models\fixfips_khatt\crnn_best.pt"
st = torch.load(p, map_location="cpu", weights_only=False)
print("top-level keys:", list(st.keys()))
for k, v in st.items():
    if k != "model":
        print(" ", k, "=", repr(v)[:400])
sd = st["model"]
print("n tensors:", len(sd))
for k, v in sd.items():
    print("  ", k, tuple(v.shape))
print("arch_version:", st.get("arch_version"))
print("use_attention:", st.get("use_attention"))
vocab = st.get("vocab")
print("vocab len:", len(vocab) if vocab else None)
if vocab:
    print("vocab repr:", json.dumps(vocab, ensure_ascii=False))
