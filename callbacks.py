from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from random import randrange
from typing import Dict
import os
import re
import numpy as np
from transformers.integrations import (
    TensorBoardCallback
)
import torch
from transformers import TrainerCallback
import scipy.cluster.hierarchy as sch


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes input and output with the provided tokenizer.
    The predicted words are colored depending on whether the prediction is correct or not.
    If the prediction is incorrect, the expected word is displayed in square brackets.

    Args:

        tokenizer: the tokenizer used to generate the dataset.

    Class Attributes:

        COLOR_CHAR (Dict): terminal colors used to produced colored string
    """

    COLOR_CHAR = {}

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer

    def on_evaluate(
        self,
        *args,
        model=None,
        eval_dataloader: torch.utils.data.DataLoader = None,
        **kwargs
    ):
        """Method called when evaluating the model. Only the needed kwargs are unpacked.

        Args:
            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            examples = self.pick_random_example(eval_dataloader)
            labels = self.tokenizer.decode(examples['labels'][0], skip_special_tokens=True)
            outputs = model.generate(examples['input_ids'])
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            inputs = self.tokenizer.decode(examples['input_ids'][0], skip_special_tokens=True)

        self.to_console(inputs, pred, labels)

    @staticmethod
    def pick_random_example(dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        length = len(dataloader.dataset)
        dataset = dataloader.dataset
        rand_example_idx = randrange(length)
        batch = dataloader.collate_fn([dataset[rand_example_idx]])  # batch with a single random example
        inputs = {}
        for k, v in batch.items():
            inputs[k] = v.cuda() if torch.cuda.is_available() else v
        return inputs

    def to_console(self, inputs, pred, labels):
        print(50*"*")
        print(f"{BColors.OKGREEN} Example of prediction in an evaluation sentence{BColors.ENDC}")
        print(f"{BColors.OKGREEN} Input: {BColors.ENDC}")
        print(inputs)
        print(f"{BColors.OKGREEN} Output: {BColors.ENDC}")
        print(pred)
        print(f"{BColors.OKGREEN} Expected output: {BColors.ENDC}")
        print(labels)
        print(50*"*")


class MyTensorBoardCallback(TensorBoardCallback):
    """Display log and metrics. Modified to plot losses together and to plot supp_data items
    passed in the model output. Also looks for logs elements with  _img_ in keys to display as images."""

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = self._rewrite_logs(logs)
            for main_tag, val in logs.items():
                if main_tag.startswith("images"):
                    val = np.array(val)
                    val = self._cluster_corr(val)
                    val = torch.from_numpy(val)  # re-tensorify
                    if val.dim() < 3:
                        val = val.unsqueeze(0)
                    val = val - val.min()
                    val = val / val.max()
                    val = 1.0 - val  # invert: high correl black, low correl white
                    val = torch.cat([val] * 3, 0)  # format C x H x W
                    self.tb_writer.add_image("images", val, state.global_step)
                else:
                    # assume a scalar
                    self.tb_writer.add_scalars(main_tag, val, state.global_step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None

    @staticmethod
    def _rewrite_logs(d):
        """
        group into 'losses' eval/loss and train/loss into with the loss breakdown provided as supp_data
        """
        new_d = {}
        for k, v in d.items():
            if k == "loss":
                new_d["losses/main_loss"] = {"train": v}
            elif k == "eval_loss":
                new_d["losses/main_loss"] = {"eval": v}
            else:
                supp_data = re.search(r"^(.*)_supp_data_(.*)", k)  # could use nge lookahead (?!img)
                img = re.search(r"^(.*)_img_(.*)", k)  # for example as in  "eval_supp_data_img_correl"
                if supp_data is not None and img is None:
                    main_tag = f"losses/{supp_data.group(1)}_supp"
                    scalar_tag = supp_data.group(2)
                    if main_tag not in new_d:
                        new_d[main_tag] = {}
                    new_d[main_tag][scalar_tag] = v
                elif img is not None:
                    main_tag = f"images/{img.group(1)}_{img.group(2)}"
                    if main_tag not in new_d:
                        new_d[main_tag] = {}
                    new_d[main_tag] = v
                else:
                    main_tag = f"other_data/{k}"
                    if main_tag not in new_d:
                        new_d[main_tag] = {}
                    new_d[main_tag][k] = v
        return new_d

    @staticmethod
    def _cluster_corr(corr_array, inplace=False):
        """
        Rearranges the correlation matrix, corr_array, so that groups of highly
        correlated variables are next to eachother

        Parameters
        ----------
        corr_array : pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix with the columns and rows rearranged
        """
        pairwise_distances = sch.distance.pdist(corr_array)
        linkage = sch.linkage(pairwise_distances, method='complete')
        cluster_distance_threshold = pairwise_distances.max() / 2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                            criterion='distance')
        idx = np.argsort(idx_to_cluster_array)

        if not inplace:
            corr_array = corr_array.copy()

        return corr_array[idx, :][:, idx]


