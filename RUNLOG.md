## Run 1 - baseline
Changed: nothing (starter defaults).
Dev bpb: 2.3718
Conclusion: constant LR, plain Adam, no warmup/clip/weight-decay, naive init, no tying. Loss still falling at step 2000 — schedule + LR are the easy win.

## Run 2 - AdamW + warmup + cosine LR + grad clip
Changed: optimizer Adam->AdamW(wd=0.1), added 100-step warmup + cosine decay, peak lr 6e-4, grad clip 1.0.
Dev bpb: 2.3593 (from 2.3718)
Conclusion: modest improvement. LR schedule alone won't fix the real bottleneck: byte tokenizer wastes budget on Hindi (3 bytes/char). Next: BPE tokenizer.
EOF

## Run 3 - BPE tokenizer (vocab 384) + weight tying + block_size 192
Changed: byte tokenizer -> BPE trained on first 400KB of corpus; tie_weights=True; init std 0.05->0.02; block_size 128->192.
Dev bpb: 2.4102 (worse than Run 2's 2.3593)
Conclusion: bpb got worse. Suspect cause: BPE merges trained on only a 400KB sample may not represent the full corpus well (if that sample is English-heavy and Hindi appears later, merges won't cover Hindi patterns properly). Also changed 3 things at once — can't isolate which hurt.
EOF
