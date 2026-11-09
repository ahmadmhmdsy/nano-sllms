"""Pre-registration param check (repo rule R5) for the option-(D) OCR architecture."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arabic_ocr" / "models" / "muharaf_ocr"))

from submission_code.torch_models import CNNTransformerCTC  # noqa: E402

CANDIDATES = [
    (256, 4, 8, 1024),   # the current 4.25M teacher, for reference
    (320, 6, 8, 1280),
    (384, 6, 8, 1536),
    (384, 8, 8, 1536),
    (384, 8, 12, 1536),
    (448, 6, 8, 1792),
    (448, 8, 8, 1792),
    (512, 4, 8, 2048),
    (512, 6, 8, 2048),
]
for d, L, H, F in CANDIDATES:
    m = CNNTransformerCTC(num_classes=166, d_model=d, num_encoder_layers=L,
                          num_heads=H, dim_feedforward=F, dropout=0.1,
                          height_pooling="attention")
    n = sum(p.numel() for p in m.parameters())
    print("d=%-4d layers=%-2d heads=%-2d ffn=%-5d -> %10s params  (%.2fx teacher)"
          % (d, L, H, F, format(n, ","), n / 4245671.0))
