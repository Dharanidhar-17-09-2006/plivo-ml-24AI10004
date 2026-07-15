My best configuration achieved a dev BPB of 2.3593 by optimizing the learning 
schedule of the 1.3M parameter baseline. I replaced the default optimizer 
with AdamW and implemented a 100-step linear warmup followed by a cosine 
learning rate decay. I set gradient clipping to 1.0 to ensure stability 
during the 2,000-step training window. I experimented with BPE tokenization 
to handle Devanagari script more efficiently but found it regressed 
performance to 2.3821 BPB. This suggested that a 2,000-step budget is 
insufficient for the larger vocabulary of a BPE tokenizer to converge 
effectively. Byte-level tokenization proved superior for this speedrun 
because the smaller 256-token vocabulary allows for more frequent updates 
per embedding. My attempts to utilize weight tying and a lower initialization 
standard deviation of 0.02 resulted in a further regression to 2.556 BPB. 
I concluded that the baseline architecture was already well-tuned for its 
size, making the learning rate schedule the most impactful lever. My final 
model prioritizes rapid convergence over architectural complexity to stay 
within the strict step limit. I chose all final hyperparameters based on 
their ability to minimize loss consistently across the full training corpus.
