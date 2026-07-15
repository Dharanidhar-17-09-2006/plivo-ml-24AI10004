## Run 1 - baseline
Changed: nothing (starter defaults).
Dev bpb: 2.3718
Conclusion: constant LR, plain Adam, no warmup/clip/weight-decay, naive init, no tying. Loss still falling at step 2000 — schedule + LR are the easy win.

{"bpb": 2.3718, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}

## Run 2 - AdamW + warmup + cosine LR + grad clip(Best)
Changed: optimizer Adam->AdamW(wd=0.1), added 100-step warmup + cosine decay, peak lr 6e-4, grad clip 1.0.
Dev bpb: 2.3593 (from 2.3718)
Conclusion: modest improvement. LR schedule alone won't fix the real bottleneck: byte tokenizer wastes budget on Hindi (3 bytes/char). Next: BPE tokenizer.

corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
model: 1,339,840 params
step     1  loss 5.6475  (174 ms/step)
step   100  loss 3.2410  (66 ms/step)
step   200  loss 2.2285  (67 ms/step)
step   300  loss 2.1738  (68 ms/step)
step   400  loss 2.1609  (68 ms/step)
step   500  loss 2.1555  (68 ms/step)
step   600  loss 2.0764  (68 ms/step)
step   700  loss 2.0276  (67 ms/step)
step   800  loss 1.9750  (67 ms/step)
step   900  loss 1.9776  (67 ms/step)
step  1000  loss 1.8913  (67 ms/step)
step  1100  loss 1.8587  (67 ms/step)
step  1200  loss 1.8400  (67 ms/step)
step  1300  loss 1.7931  (67 ms/step)
step  1400  loss 1.7443  (67 ms/step)
step  1500  loss 1.7342  (67 ms/step)
step  1600  loss 1.7284  (66 ms/step)
step  1700  loss 1.7417  (66 ms/step)
step  1800  loss 1.7020  (66 ms/step)
step  1900  loss 1.7186  (66 ms/step)
step  2000  loss 1.7143  (66 ms/step)
saved ckpt.pt  (132s total)

{"bpb": 2.3593, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}

## Run 3 - BPE tokenizer (vocab 384) + weight tying + block_size 192
Changed: byte tokenizer -> BPE trained on first 400KB of corpus; tie_weights=True; init std 0.05->0.02; block_size 128->192.
Dev bpb: 2.4102 (worse than Run 2's 2.3593)
Conclusion: bpb got worse. Suspect cause: BPE merges trained on only a 400KB sample may not represent the full corpus well (if that sample is English-heavy and Hindi appears later, merges won't cover Hindi patterns properly). Also changed 3 things at once — can't isolate which hurt.

{"bpb": 2.4102, "n_params": 1329600, "steps": 2000, "tokens_in_eval": 92921, "tokens_scored": 92920}

## Run 4 - BPE (vocab 384, sampled across whole corpus) + weight tying + block_size 192
Changed: BPE merges retrained on samples spread across full corpus instead of just the start.
Dev bpb: 2.3821 (better than Run 3's 2.4102, still worse than Run 2's 2.3593 byte tokenizer)
Conclusion: BPE isn't paying off under this fixed 2000-step budget. Fewer, larger-vocab tokens mean each optimizer step covers more content, but the model has less time to learn a bigger embedding/output table from scratch. Byte tokenizer's small vocab (256) converges faster within 2000 steps even though sequences are longer. Reverting to byte tokenizer, keeping the AdamW/schedule/init/tying wins from Run 2.

{"bpb": 2.3821, "n_params": 1329600, "steps": 2000, "tokens_in_eval": 87912, "tokens_scored": 87911}

## Run 5 - byte tokenizer + weight tying + init std 0.02
Changed: reverted to byte tokenizer, kept tie_weights=True, init std 0.02, block_size 128.
Dev bpb: 2.556 (worse than Run 2's 2.3593)
Conclusion: tying+init hurt here. Reverting both, keeping only Run 2's AdamW+schedule as best.

{"bpb": 2.556, "n_params": 1298880, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}
