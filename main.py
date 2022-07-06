# -*- coding: utf-8 -*-
import logging
from blurb_benchmark import BlurbIOBToSeq2Seq
from seq2seq.blurb_trainer import BlurbTrainer
from data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq
from seq2seq.trainer import SodaSeq2SeqTrainer
from data_utils import get_label_entity_pairs, get_control_measure_exp_mentions, get_labelled_data
from metrics import ClassificationSeq2Seq, BlurbMetrics
logger = logging.getLogger('seq2seq.main')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bb = BlurbIOBToSeq2Seq(dataset_name = "JNLPBA",
                 model = "facebook/bart-base",
                 separator = "|separator|",
                 label_mode = "full_text",
                 output = "gpt", #gpt, hf_csv, None,
                 closing_token_outputs = "[END]", # Needed for GPT-3 API
                 class_start = "_class_***", # Needed for GPT-3 API
                 class_end = "***_class_" # Needed for GPT-3 API
)

    # bb = BlurbTrainer(dataset_name="BC5CDR-chem-IOB",
    #                  task_name="Tag entities",
    #                  model="facebook/bart-large",
    #                  from_pretrained="facebook/bart-large",
    #                  from_local_checkpoint=None,
    #                  base_model="facebook/bart-large",
    #                  # TOKENIZER PARAMETERS
    #                  max_input_length=512,
    #                  max_target_length=512,
    #                  # MODEL PARAMETERS
    #                  model_param=ModelConfigSeq2Seq(),
    #                  # TRAINING PARAMETERS
    #                  training_args=TrainingArgumentsSeq2Seq(),
    #                  separator=",and ",
    #                  label_mode="labels_only"
    #                 )
    # bm = BlurbMetrics("./seq2seq_models/facebook___bart-large_full-text.csv", "###separator###", 'full-text')
    # bm()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
