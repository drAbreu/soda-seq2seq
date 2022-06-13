
Training the model from a model in the 🤗 Hub.

```bash
  python -m cli.seq2seq_train ./data/sd-seq2seq-clean.csv \
  --from_pretrained facebook/bart-base \
  --task "Causal hypothesis: " \
  --delimiter "###tt9HHSlkWoUM###" \
  --skip_lines 0 \
  --eval_steps 500 \
  --logging_steps 50 \
  --num_train_epochs 10 \
  --lr_scheduler_type 'cosine' \
  --warmup_steps 5000 
```

Training the model beginning from a locally stored checkpoint.

```bash
  python -m cli.seq2seq_train ./data/sd-seq2seq-clean.csv \
  --from_local_checkpoint ./seq_2seq_models/checkpoint-5000 \
  --base_model facebook/bart-base \
  --task "Causal hypothesis: " \
  --delimiter "###tt9HHSlkWoUM###" \
  --skip_lines 0 \
  --eval_steps 500 \
  --logging_steps 50 \
  --num_train_epochs 10 \
  --lr_scheduler_type 'cosine' \
  --warmup_steps 5000 
```
