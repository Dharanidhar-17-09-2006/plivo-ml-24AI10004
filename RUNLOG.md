## Run 1 - baseline
Changed: nothing (starter defaults).
Dev bpb: 2.3718
Conclusion: constant LR, plain Adam, no warmup/clip/weight-decay, naive init, no tying. Loss still falling at step 2000 — schedule + LR are the easy win.

## Run 2 - AdamW + warmup + cosine LR + grad clip
Changed: optimizer Adam->AdamW(wd=0.1), added 100-step warmup + cosine decay, peak lr 6e-4, grad clip 1.0.
Dev bpb: 2.3593 (from 2.3718)
Conclusion: modest improvement. LR schedule alone won't fix the real bottleneck: byte tokenizer wastes budget on Hindi (3 bytes/char). Next: BPE tokenizer.
EOF
