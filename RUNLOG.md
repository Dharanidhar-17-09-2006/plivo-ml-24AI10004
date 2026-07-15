## Run 1 - baseline
Changed: nothing (starter defaults).
Dev bpb: 2.3718
Conclusion: constant LR, plain Adam, no warmup/clip/weight-decay, naive init, no tying. Loss still falling at step 2000 — schedule + LR are the easy win.
