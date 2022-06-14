`.env` file for this repo:

```properties
    SEQ2SEQ_MODEL_PATH="./seq2seq_models"
    CACHE="./cache"
    RUNS_DIR="./runs"
    RUNTIME=""
```

The way to install this repo is as with any other `venv` repo.

```shell
    git clone https://github.com/drAbreu/soda-seq2seq.git
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
```

Training the model from a model in the 🤗 Hub.

```shell
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

```shell
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
