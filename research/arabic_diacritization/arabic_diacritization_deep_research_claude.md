Existing open-source projects (ranked by relevance)



1\. CATT (Character-based Arabic Tashkeel Transformer) — abjadai/catt, most modern and currently the strongest open baseline.



Character-level Transformer, two variants: Encoder-Only (fast) and Encoder-Decoder (more accurate)

Ships pretrained checkpoints, an ONNX export path, a training script, and a downloadable dataset — Apache-2.0 licensed (was CC-BY-NC, now fully open for commercial use)

Uses a custom char-level BERT pretraining step before fine-tuning on diacritization — this "pretrain then fine-tune" pattern is exactly the kind of thing you could reuse instead of training from zero

This is your best starting point to fork/build on.



2\. Shakkelha (AliOsm/shakkelha) — the reference implementation behind the widely-cited Fadel et al. 2019 paper (RNN + BNG embeddings). Good baseline/benchmark reference, older architecture (LSTM-based, pre-Transformer), DER \~2.88%.



3\. libtashkeel (mush42/libtashkeel) — Rust/C++/WASM/Python cross-platform inference library, models trained on MSA data. Good reference if you eventually want a small, embeddable production runtime rather than a Python-only model.



4\. Older/rule-based: Mishkal, Shakkala (original), Farasa — mostly rule-based or older neural, useful only as baselines to beat, DER much worse (\~13.78% for Mishkal vs \~2.88% neural).



5\. Recent research direction worth copying: "Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need" (2024) — frames diacritization as token/character classification on top of a pretrained Arabic LM (like AraBERT), rather than training a seq2seq model from scratch. This is a strong candidate for your "smaller but better quality" goal since you inherit language understanding from a pretrained encoder.



Datasets available

Dataset	Size	Domain	Notes

Tashkeela (Zerrouki \& Balla)	\~75M words, 97 books	Classical Arabic (Islamic texts, 98.85%)	The dominant corpus; huge but domain-skewed toward classical/religious text

Fadel et al. 2019 cleaned Tashkeela subset	2.3M words / 55K lines	Classical, cleaned	The standard benchmark split (train/valid/test) — consists of about 2.3M words spread over 55K lines with 2,103K words for training, 102K for validation, and 107K for testing. Best place to start for a "small but high quality" set

CATT dataset	Bundled with repo	Mixed	Already cleaned/prepped for their tokenizer/pipeline — reuse this if you fork CATT

Penn Arabic Treebank (ATB)	Smaller	MSA (news)	Standard MSA benchmark, alongside WikiNews

WikiNews (Darwish et al.)	Small	MSA (news)	Standard MSA benchmark — good for modern/news-style text since Tashkeela is classical-heavy

APCD / APCD2 (Yousef et al.)	Large	Poetry	Useful only if you care about classical poetry meter



Practical recommendation: combine the cleaned Tashkeela subset (Fadel et al.) for volume with ATB + WikiNews for MSA/news-style coverage, since there is a clear domain gap between the Tashkeela and Wiki News data, with every diacritic showing a 4-45% accuracy drop when a Tashkeela-trained model is tested on news text. If your product's real-world input is more like modern news/social text than classical religious text, you cannot skip this domain-mixing step. 

Diva-portal



Build vs. build-from-scratch verdict

Don't start from zero. Fork CATT's character-level Transformer architecture and training pipeline — it's modern, Apache-2.0, already has ONNX export (useful for mobile/production later, which fits your stack), and ships a working char-BERT pretraining recipe.

Your "smaller but better" angle: use CATT's architecture but curate a cleaner, more balanced training mix (Tashkeela subset + MSA news + your own in-domain text) rather than raw Tashkeela — quality-over-quantity is a legitimate and underexplored angle here.

On your 1,000-token context plan



Given Tashkeela's average is 42.06 words per line and 3.97 characters per word, a 1,000-token window (character-level tokens) comfortably covers several sentences — reasonable for a first training pass. Scaling later just means longer max\_seq\_len (CATT already trains at max\_seq\_len=1024) and more compute, not an architecture change. 

arxiv



Want me to go one level deeper — e.g. pull the CATT paper's exact architecture/hyperparameters, or scope out a concrete phase-1 training plan (data prep, tokenizer, model size, training script) tailored to your 1,000-token start?



Here's the deep dive — architecture, data pipeline, and a concrete phase-1 plan sized to your 1,000-token start.



CATT architecture (the one to clone/build on)



From the paper (Alasmary et al., 2024, ArabicNLP):



Stage 1 — Pretrain a character-level BERT (MLM)



Trained with standard masked-language-modeling on raw Arabic character sequences (no diacritics needed for this stage — huge amount of unlabeled Arabic text works)

Max sequence length during pretraining: 512 characters (samples truncated at the last space if longer, to preserve whole-word boundaries)

This gives the model a general sense of Arabic character/morphology patterns before it ever sees a diacritic



Stage 2 — Fine-tune two variants on the diacritization task



Encoder-Only (EO): 6 layers, 512 d\_model, 16 attention heads — each output character position directly classified into a diacritic class. Faster inference, recommended for production/latency-sensitive use.

Encoder-Decoder (ED): same encoder initialized from the char-BERT, plus a decoder — generates diacritics more accurately but slower. Recommended when accuracy matters more than speed.

Max sequence length at fine-tune time: 1024 characters (this is your practical ceiling reference point — your 1,000-token plan lines up almost exactly with theirs)

Character-level, not word/subword-level tokenization — this matters a lot for Arabic because diacritics attach per-letter, not per-token, so a word-piece tokenizer would need extra machinery to map predictions back to letters. Character-level avoids that entirely.



Stage 3 — Noisy Student self-training



Once you have a decent fine-tuned model, run it over \~1M unlabeled sentences to generate pseudo-diacritized labels, then mix those pseudo-labels back into training and retrain. This is what pushed CATT past everything else, including GPT-4-turbo on their benchmark (a relative DER improvement of 9.36% over GPT-4-turbo on the CATT dataset). This step is what you'd want for your "smaller but better" angle — it's a way to get more effective training signal without collecting more manually-labeled data.



Benchmarks it beats: state-of-the-art results with relative DER improvements of 30.83% on WikiNews and 35.21% on their own CATT benchmark versus prior best models.



Concrete Phase 1 plan (sized to 1,000 tokens)



1\. Data preparation



Base corpus: cleaned Tashkeela subset (Fadel et al., 2019) — 2.3M words / 55K lines, already split train/valid/test

Add domain diversity: WikiNews + ATB for MSA/news register, since Tashkeela is \~99% classical/religious text and a model trained only on it degrades badly on modern text

Chunk everything into ≤1,000-character windows, splitting at sentence/word boundaries (not mid-word) — mirror CATT's "truncate at last space" trick

Build a small, fixed character vocabulary: Arabic letters + a "no diacritic" class + the 8 core diacritic marks (fatha, damma, kasra, sukun, shadda, tanwin variants, and combinations) — this is a closed, small label set, which is exactly why character-level classification works so well here



2\. Tokenizer



Character-level, not BPE. Two parallel streams: (a) input characters with diacritics stripped, (b) target diacritic label per character position. Handle the "already diacritized" case in your prompt by just stripping any existing diacritics as a preprocessing step before feeding the model — this makes "may or may not be diacritized" input a non-issue architecturally.



3\. Model size for a first run



Don't clone CATT's exact size (6 layers/512 dim) on day one — start smaller (e.g. 4 layers, 256–384 dim) at your 1,000-token context to get the full pipeline working end-to-end fast, then scale dimensions/layers once the pipeline and eval harness are validated. Scaling context length later (1,000 → 2,000+) is cheap; scaling your data-cleaning and eval pipeline later is not, so get those right first.



4\. Evaluation



Use the standard two metrics from this literature: DER (Diacritic Error Rate — % of misclassified characters) and WER (% of words with ≥1 misclassified character). Every paper in this space reports both, so you'll want them for comparability.

Hold out WikiNews as your "modern text" test set specifically, separate from the Tashkeela-derived test set, so you can see the classical/MSA domain gap directly rather than getting a misleadingly good number from an all-classical test set.



5\. Build-vs-fork decision for your repo



Practically: fork abjadai/catt (Apache-2.0, so no licensing issue), strip it down to a smaller config, swap in your mixed dataset, and treat their char-BERT pretraining script as optional (skip it for v1, add it in a later phase once you want to push accuracy further via self-training).



Want me to sketch the actual label schema (the full diacritic class list mapped to Unicode combining marks) and a first-pass PyTorch model skeleton at the \~1,000-token scale?



go



Thought for 2s

Designing a diacritic labeling scheme and neural model skeleton.

1\. Diacritic label schema



Arabic diacritics you need to cover (Unicode combining marks, range U+064B–U+0652 plus shadda combos):



python

\# Arabic diacritics (harakat) — Unicode combining marks

FATHA      = '\\u064E'  # َ

DAMMA      = '\\u064F'  # ُ

KASRA      = '\\u0650'  # ِ

SUKUN      = '\\u0652'  # ْ

SHADDA     = '\\u0651'  # ّ

FATHATAN   = '\\u064B'  # ً  (tanwin fath)

DAMMATAN   = '\\u064C'  # ٌ  (tanwin damm)

KASRATAN   = '\\u064D'  # ٍ  (tanwin kasr)



\# A letter can carry 0, 1, or 2 diacritics (shadda + one vowel mark, e.g. الشَّمْس)

\# Rather than predicting each mark independently, treat it as single-label

\# classification over the \*combined\* set — this is what Fadel et al. and CATT do.

DIACRITIC\_CLASSES = \[

&#x20;   '',                        # no diacritic

&#x20;   FATHA, DAMMA, KASRA, SUKUN, FATHATAN, DAMMATAN, KASRATAN,

&#x20;   SHADDA,                                    # shadda alone

&#x20;   SHADDA + FATHA, SHADDA + DAMMA, SHADDA + KASRA,

&#x20;   SHADDA + FATHATAN, SHADDA + DAMMATAN, SHADDA + KASRATAN,

]

\# 15 classes total — a small, closed softmax head. This is why char-level

\# classification (not generation) is the right framing: it's a per-position

\# 15-way classification problem, not open-ended text generation.



label2id = {lab: i for i, lab in enumerate(DIACRITIC\_CLASSES)}

id2label = {i: lab for lab, i in label2id.items()}



Preprocessing rule: strip any existing diacritics from input first (regex over U+064B–U+0652), always predict fresh. This is what makes "input may or may not already be diacritized" a non-problem — you normalize to undiacritized text before the model ever sees it.



python

import re

DIACRITIC\_PATTERN = re.compile('\[\\u064B-\\u0652]')



def strip\_diacritics(text: str) -> str:

&#x20;   return DIACRITIC\_PATTERN.sub('', text)



def build\_training\_pair(diacritized\_line: str):

&#x20;   """Split a fully-diacritized line into (base\_chars, label\_ids)."""

&#x20;   chars, labels = \[], \[]

&#x20;   i = 0

&#x20;   while i < len(diacritized\_line):

&#x20;       ch = diacritized\_line\[i]

&#x20;       if DIACRITIC\_PATTERN.match(ch):

&#x20;           i += 1

&#x20;           continue  # shouldn't happen if we start from base char

&#x20;       # look ahead for attached diacritic(s)

&#x20;       j = i + 1

&#x20;       mark = ''

&#x20;       while j < len(diacritized\_line) and DIACRITIC\_PATTERN.match(diacritized\_line\[j]):

&#x20;           mark += diacritized\_line\[j]

&#x20;           j += 1

&#x20;       # normalize order: shadda always before vowel mark in our class set

&#x20;       if len(mark) == 2 and mark\[0] != SHADDA:

&#x20;           mark = mark\[1] + mark\[0]

&#x20;       chars.append(ch)

&#x20;       labels.append(label2id.get(mark, 0))

&#x20;       i = j

&#x20;   return chars, labels

2\. Vocabulary (character-level)

python

ARABIC\_LETTERS = list('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي')

PUNCT\_AND\_DIGITS = list(' .,؛؟!()0123456789-\\n')

SPECIAL\_TOKENS = \['<pad>', '<unk>']



VOCAB = SPECIAL\_TOKENS + ARABIC\_LETTERS + PUNCT\_AND\_DIGITS

char2id = {c: i for i, c in enumerate(VOCAB)}

PAD\_ID = char2id\['<pad>']

UNK\_ID = char2id\['<unk>']



def encode\_chars(chars, max\_len=1000):

&#x20;   ids = \[char2id.get(c, UNK\_ID) for c in chars]\[:max\_len]

&#x20;   ids += \[PAD\_ID] \* (max\_len - len(ids))

&#x20;   return ids



Vocab size lands around \~70 — tiny compared to any subword vocab, which is part of why this task is tractable at small model scale.



3\. Model skeleton (PyTorch, sized for 1,000-token context)



Encoder-only classification head — start here before attempting the encoder-decoder variant:



python

import torch

import torch.nn as nn



class CharDiacritizer(nn.Module):

&#x20;   def \_\_init\_\_(

&#x20;       self,

&#x20;       vocab\_size: int,

&#x20;       num\_labels: int = len(DIACRITIC\_CLASSES),

&#x20;       d\_model: int = 256,      # start smaller than CATT's 512 for v1

&#x20;       n\_heads: int = 8,

&#x20;       n\_layers: int = 4,       # start smaller than CATT's 6 for v1

&#x20;       max\_seq\_len: int = 1000,

&#x20;       dropout: float = 0.1,

&#x20;       pad\_id: int = 0,

&#x20;   ):

&#x20;       super().\_\_init\_\_()

&#x20;       self.pad\_id = pad\_id

&#x20;       self.char\_embed = nn.Embedding(vocab\_size, d\_model, padding\_idx=pad\_id)

&#x20;       self.pos\_embed = nn.Embedding(max\_seq\_len, d\_model)



&#x20;       encoder\_layer = nn.TransformerEncoderLayer(

&#x20;           d\_model=d\_model,

&#x20;           nhead=n\_heads,

&#x20;           dim\_feedforward=d\_model \* 4,

&#x20;           dropout=dropout,

&#x20;           batch\_first=True,

&#x20;           activation='gelu',

&#x20;       )

&#x20;       self.encoder = nn.TransformerEncoder(encoder\_layer, num\_layers=n\_layers)

&#x20;       self.norm = nn.LayerNorm(d\_model)

&#x20;       self.classifier = nn.Linear(d\_model, num\_labels)



&#x20;   def forward(self, input\_ids: torch.Tensor) -> torch.Tensor:

&#x20;       # input\_ids: (batch, seq\_len)

&#x20;       B, L = input\_ids.shape

&#x20;       positions = torch.arange(L, device=input\_ids.device).unsqueeze(0).expand(B, L)

&#x20;       x = self.char\_embed(input\_ids) + self.pos\_embed(positions)



&#x20;       pad\_mask = input\_ids.eq(self.pad\_id)  # (batch, seq\_len), True = ignore

&#x20;       x = self.encoder(x, src\_key\_padding\_mask=pad\_mask)

&#x20;       x = self.norm(x)

&#x20;       return self.classifier(x)  # (batch, seq\_len, num\_labels)



Training loop essentials:



python

def compute\_loss(logits, labels, pad\_mask):

&#x20;   # logits: (B, L, C), labels: (B, L)

&#x20;   loss\_fct = nn.CrossEntropyLoss(ignore\_index=-100)

&#x20;   labels = labels.masked\_fill(pad\_mask, -100)  # don't penalize padding

&#x20;   return loss\_fct(logits.view(-1, logits.size(-1)), labels.view(-1))



\# Optimizer: AdamW, lr \~3e-4 with warmup, matches typical small-transformer setups

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight\_decay=0.01)

4\. DER/WER eval (so you can compare against published numbers)

python

def diacritic\_error\_rate(pred\_ids, gold\_ids, pad\_mask):

&#x20;   """% of non-pad characters with wrong diacritic class."""

&#x20;   mismatches = (pred\_ids != gold\_ids) \& (\~pad\_mask)

&#x20;   return mismatches.sum().item() / (\~pad\_mask).sum().item()



def word\_error\_rate(pred\_ids, gold\_ids, pad\_mask, is\_space\_mask):

&#x20;   """% of words containing >=1 mismatched character."""

&#x20;   # split by is\_space\_mask into word spans, flag a word if any char mismatches

&#x20;   ...  # implement by grouping char indices between space positions

Where this sits relative to CATT



This skeleton is deliberately CATT-shaped but scaled down (4 layers/256-dim vs their 6/512, no char-BERT pretrain stage yet, no Noisy-Student self-training yet) — get this training and hitting a reasonable DER on the Tashkeela-cleaned test set first, then layer in:



Char-BERT MLM pretraining on unlabeled Arabic text (improves generalization, especially to MSA/news)

Scale to their 6-layer/512-dim/1024-context config

Noisy-Student pseudo-labeling round

