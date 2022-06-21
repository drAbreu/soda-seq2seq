from transformers import TrainingArguments, PretrainedConfig, SchedulerType, IntervalStrategy
from dataclasses import dataclass, field
from seq2seq import SEQ2SEQ_MODEL_PATH
from typing import Any, Dict, List, Optional, NamedTuple, Union, Tuple
import numpy as np

@dataclass
class TrainingArgumentsSeq2Seq(TrainingArguments):
    output_dir: str = field(default=SEQ2SEQ_MODEL_PATH)
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=False)
    evaluation_strategy: IntervalStrategy = field(default="steps")
    prediction_loss_only: bool = field(default=False)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    eval_accumulation_steps: Optional[int] = field(default=None)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=-1)
    lr_scheduler_type: SchedulerType = field(default="constant")
    warmup_ratio: float = field(default=0.0)
    warmup_steps: int = field(default=0)
    log_level: Optional[str] = field(default="passive")
    logging_dir: Optional[str] = field(default=None)
    logging_strategy: IntervalStrategy = field(default="steps")
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=50)
    logging_nan_inf_filter: str = field(default=True)
    save_strategy: IntervalStrategy = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=None)
    seed: int = field(default=42)
    dataloader_drop_last: bool = field(default=False)
    eval_steps: int = field(default=2500)
    dataloader_num_workers: int = field(default=0)
    run_name: Optional[str] = field(default="Seq2Seq")
    disable_tqdm: Optional[bool] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=False)
    metric_for_best_model: Optional[str] = field(default=None)
    greater_is_better: Optional[bool] = field(default=None)
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    report_to: Optional[List[str]] = field(default='tensorboard')
    ddp_find_unused_parameters: Optional[bool] = field(default=None)
    push_to_hub: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    hub_model_id: str = field(default=None)
    # hub_strategy: HubStrategy = field(default="every_save")
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(default=False)
    generation_max_length: int = field(default=128)
    generation_num_beams: int = field(default=2)
    predict_with_generate: bool = field(default=True)


@dataclass
class ModelConfigSeq2Seq(PretrainedConfig):
    max_length: int = field(default=128)
    min_length: int = field(default=32)
    early_stopping: bool = field(default=False)
    num_beams: int = field(default=1)
    num_beam_groups: int = field(default=1)
    diversity_penalty: float = field(default=0.0)
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=1.)
    num_return_sequences: int = field(default=1)
    output_scores: bool = field(default=False)
    length_penalty: float = field(default=1.)
    repetition_penalty: float = field(default=2.)
    no_repeat_ngram_size: int = field(default=5)
    return_dict_in_generate: bool = field(default=False)

