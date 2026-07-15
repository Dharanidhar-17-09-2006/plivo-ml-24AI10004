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

cat >> RUNLOG.md << 'EOF'

## Run 4 - BPE (vocab 384, sampled across whole corpus) + weight tying + block_size 192
Changed: BPE merges retrained on samples spread across full corpus instead of just the start.
Dev bpb: 2.3821 (better than Run 3's 2.4102, still worse than Run 2's 2.3593 byte tokenizer)
Conclusion: BPE isn't paying off under this fixed 2000-step budget. Fewer, larger-vocab tokens mean each optimizer step covers more content, but the model has less time to learn a bigger embedding/output table from scratch. Byte tokenizer's small vocab (256) converges faster within 2000 steps even though sequences are longer. Reverting to byte tokenizer, keeping the AdamW/schedule/init/tying wins from Run 2.
EOF
