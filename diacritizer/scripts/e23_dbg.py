from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, 'diacritizer/src')
import eval_der as ED
marks_of = ED.marks_of
base = Path(r'runs\diac\e23d_bilstm_lr1e-3\gate_probe')
p = (base / 'step499_fadel_test_pred.txt').read_text(encoding='utf-8').splitlines()
r = (base / 'step499_fadel_test_pred.ref.txt').read_text(encoding='utf-8').splitlines()
pw = " ".join(p).split()
rw = " ".join(r).split()
cnt = Counter()
for w in pw:
    m = marks_of(w)
    cnt['bare' if not m else m[0].encode('unicode_escape').decode()[:6]] += 1
cr = Counter()
for w in rw:
    m = marks_of(w)
    cr['bare' if not m else m[0].encode('unicode_escape').decode()[:6]] += 1
for k in sorted(set(list(cnt) + list(cr))):
    print(k, 'pred', cnt.get(k, 0), 'ref', cr.get(k, 0))
exact = sum(1 for a, b in zip(pw[:len(rw)], rw) if marks_of(a) == marks_of(b))
print('exact matches', exact, '/', len(rw))
