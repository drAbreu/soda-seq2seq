# -*- coding: utf-8 -*-
import logging
from blurb_benchmark import BlurbBenchmark
from seq2seq.blurb_trainer import BlurbTrainer
from data_classes import ModelConfigSeq2Seq, TrainingArgumentsSeq2Seq
from seq2seq.trainer import SodaSeq2SeqTrainer
from data_utils import get_label_entity_pairs, get_control_measure_exp_mentions, get_labelled_data
from metrics import ClassificationSeq2Seq
logger = logging.getLogger('seq2seq.main')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # LABEL = ['''gene:cyclin f or gene:e2f1 or gene:ubiquitin or molecule:ly2603618 or protein:chk1 was tested for its influence on protein:e2f1 or protein:ub by immunoblotted, immunoprecipitated "''',
    #          '''gene:igfbp1a was tested for its influence on cell:-cell or organism:zebrafish by fluorescence microscopy''',
    #          '''gene:atg5 was tested for its influence on tissue:hyperplastic lesions or tissue:lung or tissue:tumour by haematoxylin and eosin staining''']
    # OUTPUT = ['''gene:cyclin f or gene:ubiquitin or molecule:chk1i was tested for its influence on protein:e2f1 or protein:ubiquitin by immunoblotted, immunoprecipitated "''',
    #           '''gene:igfbp1a was tested for its influence on cell:-cell by regeneration assays.''',
    #           '''gene:atg5 was tested for its influence on tissue:lung by hyperplastic lesions. undefined:hyperplastic lesions or tissue:lung by morphological analysis''']
    # SEPARATORS = ['was tested for its influence', 'by']
    #
    #
    # metrics = ClassificationSeq2Seq(task="experiment")
    # metrics(LABEL, OUTPUT)
    # data_loader = "./data/sd-seq2seq-clean.csv"
    # delimiter = "###tt9HHSlkWoUM###"
    # from_pretrained = "###tt9HHSlkWoUM###"
    # task = "facebook/bart-base"
    # from_local_checkpoint = "./seq2seq_models/checkpoint-10000"
    # base_model = "t5-base"
    #
    # seq2seq = SodaSeq2SeqTrainer(data_loader,
    #                              delimiter,
    #                              from_pretrained,
    #                              task,
    #                              from_local_checkpoint=from_local_checkpoint,
    #                              base_model=base_model,
    #                              split=split,
    #                              skip_lines=skip_lines,
    #                              max_input_length=max_input_length,
    #                              max_target_length=max_target_length,
    #                              training_args=training_args,
    #                              model_param=model_args)
    bb = BlurbTrainer(dataset_name="BC5CDR-chem-IOB",
                     task_name="Tag entities",
                     model="facebook/bart-base",
                     from_pretrained="facebook/bart-base",
                     from_local_checkpoint=None,
                     base_model="facebook/bart-base",
                     # TOKENIZER PARAMETERS
                     max_input_length=512,
                     max_target_length=512,
                     # MODEL PARAMETERS
                     model_param=ModelConfigSeq2Seq(),
                     # TRAINING PARAMETERS
                     training_args=TrainingArgumentsSeq2Seq(),
                    )

    bb()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
