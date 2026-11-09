"""Pure numpy inference + per-language int8 quantization for DA-1 (no torch)."""
import numpy as np

from langid.src import features


def quantize_columns(W):
    """Per-language-column int8 quantization: Wq[:, l] ~= W[:, l] * scale[l].

    Max-abs symmetric scale per language column; worst-case reconstruction
    error per weight is max_abs[l] / 254.
    """
    W = np.asarray(W, dtype=np.float64)
    max_abs = np.max(np.abs(W), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    scale = 127.0 / max_abs
    Wq = np.clip(np.round(W * scale), -127.0, 127.0).astype(np.int8)
    return Wq, scale


class Int8LangID:
    """Loads a langid_int8.npz artifact and predicts with numpy only."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.Wq = z["W"]
        self.scale = z["scale"].astype(np.float64)
        self.bias = z["bias"].astype(np.float64)
        self.langs = [str(x) for x in z["langs"].tolist()]

    def scores(self, text: str):
        """Raw per-language scores, or (None, langs) when there are no
        letter features (abstain - never a guess)."""
        ids = np.fromiter(features.text_features(text), dtype=np.int64)
        if ids.size == 0:
            return None, self.langs
        # sum(Wq)/scale ~= sum(W): int64 accumulation avoids int8 overflow
        return self.Wq[ids].sum(axis=0, dtype=np.int64).astype(np.float64) / self.scale + self.bias, self.langs

    def predict(self, text: str, margin: float = 0.5):
        """Returns (lang, is_tie_or_abstain). Ties/abstains return (None, True)."""
        s, langs = self.scores(text)
        if s is None:
            return None, True
        order = np.argsort(s)[::-1]
        i1, i2 = int(order[0]), int(order[1])
        if s[i1] - s[i2] < margin:
            return None, True
        return langs[i1], False
