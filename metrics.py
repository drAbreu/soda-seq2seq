from typing import List
from sklearn.metrics import classification_report
from data_utils import get_labelled_data
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import textdistance
from tqdm import tqdm
from seq2seq.blurb_trainer import BlurbTrainer
import re

class ClassificationSeq2Seq:
    def __init__(self,
                task: str = 'roles',
                ner_labels: List[str] = ["gene", "protein", "molecule",
                                         "cell", "organism", "tissue", "subcellular"],
                roles_labels: List[str] = ["MEASURED_VAR", "CONTROLLED_VAR"]
                ):
        assert task in ['roles', 'ner', 'experiment'], f"""Please, choose one of {['roles', 'ner', 'experiment']}"""
        self.task = task
        self.ner_labels = ner_labels
        self.roles_labels = roles_labels

    def __call__(self,
                 predictions: List[str],
                 labels: List[str],
                 separators: List[str] = ["was tested for its influence", "by"]
                 ):
        assert len(predictions) == len(labels), """predictions and labels must have the same size"""

        if self.task == 'roles':
            cleaned_outputs = {'predictions': [], 'labels': []}
            for prediction, label in tqdm(zip(predictions, labels)):
                pred_controls, pred_measured, _ = get_labelled_data(prediction, separators=separators)
                label_controls, label_measured, _ = get_labelled_data(label, separators=separators)
                print(pred_controls, pred_measured)
                print(label_controls, label_measured)
                for label_pair in label_controls:
                    if label_pair in pred_controls:
                        cleaned_outputs['predictions'].append("CONTROLLED_VAR")
                        cleaned_outputs['labels'].append("CONTROLLED_VAR")
                        pred_controls.remove(label_pair)
                    if label_pair not in pred_controls:
                        cleaned_outputs['predictions'].append("O")
                        cleaned_outputs['labels'].append("CONTROLLED_VAR")
                for label_pair in pred_controls:
                    cleaned_outputs['predictions'].append("CONTROLLED_VAR")
                    cleaned_outputs['labels'].append("O")
                for label_pair in label_measured:
                    if label_pair in pred_measured:
                        cleaned_outputs['predictions'].append("MEASURED_VAR")
                        cleaned_outputs['labels'].append("MEASURED_VAR")
                        pred_measured.remove(label_pair)
                    if label_pair not in pred_measured:
                        cleaned_outputs['predictions'].append("O")
                        cleaned_outputs['labels'].append("MEASURED_VAR")
                for label_pair in pred_measured:
                    cleaned_outputs['predictions'].append("MEASURED_VAR")
                    cleaned_outputs['labels'].append("O")
            print(classification_report(np.array(cleaned_outputs['labels']),
                                        np.array(cleaned_outputs['predictions']),
                                        labels=self.roles_labels
                                        )
                  )
            return np.array(cleaned_outputs['labels']),np.array(cleaned_outputs['predictions'])

        elif self.task == 'ner':
            cleaned_outputs = {'predictions': [], 'labels': []}
            for prediction, label in tqdm(zip(predictions, labels)):
                pred_controls, pred_measured, _ = get_labelled_data(prediction, separators=separators)
                label_controls, label_measured, _ = get_labelled_data(label, separators=separators)
                pred_controls = list(set(pred_controls))
                pred_measured = list(set(pred_measured))
                all_preds = list(itertools.chain(pred_controls, pred_measured))
                all_labels = list(itertools.chain(label_controls, label_measured))
                for label_pair in all_labels:
                    cleaned_outputs['labels'].append(label_pair[0])
                    if label_pair in all_preds:
                        cleaned_outputs['predictions'].append(label_pair[0])
                        all_preds.remove(label_pair)
                    else:
                        cleaned_outputs['predictions'].append("O")
                for label_pair in all_preds:
                    cleaned_outputs['predictions'].append(label_pair[0])
                    cleaned_outputs['labels'].append("O")
            print(classification_report(np.array(cleaned_outputs['labels']),
                                        np.array(cleaned_outputs['predictions']),
                                        labels=self.ner_labels
                                        )
                  )
            return np.array(cleaned_outputs['labels']), np.array(cleaned_outputs['predictions'])

        elif self.task == 'experiment':
            jaccard_distance = []
            for prediction, label in tqdm(zip(predictions, labels)):
                _, _, pred_experiment = get_labelled_data(prediction, separators=separators)
                _, _, label_experiment = get_labelled_data(label, separators=separators)
                if type(pred_experiment) == list:
                    pred_experiment = " ".join(pred_experiment)
                if type(label_experiment) == list:
                    label_experiment = " ".join(label_experiment)
                jaccard_distance.append(textdistance.jaccard(
                    label_experiment.split(),
                    pred_experiment.split())
                )
            print(f"""The Average Jaccard Distance experiment strings: {np.array(jaccard_distance).mean()}""")
            return np.array(jaccard_distance).mean()
        else:
            pass


class BlurbMetrics:
    def __init__(self,
                results_file: str,
                separator: str,
                label_mode: str):
        self.results_file = results_file
        self.separator = separator
        self.predictions, self.labels = self._read_results_file()
        self.label_mode = label_mode
        self.id2label = {0: "0", 1: "B-Chemical", 2: "I-Chemical"}
        self.label2id = {"0": 0, "B-Chemical": 1, "I-Chemical": 2}
        self.tagged_entities_regex = r"([B-I]-\S+)\:(\S+)"

    def __call__(self):
        if self.label_mode == "full-text":
            counter = 0
            predictions_list, labels_list = [], []
            for p, l in zip(self.predictions, self.labels):
                result = self._get_single_result_full_text(p.split(".")[0], l.split(".")[0])
                if result == ([], []):
                    counter += 1
                else:
                    predictions_list.extend(result[0])
                    labels_list.extend(result[1])
            print(f"Missmatches {counter} of {len(self.predictions)} = {100 * counter / len(self.predictions)}")
            print(np.array(labels_list).shape)
            print(np.array(predictions_list).shape)
            print(classification_report(np.array(labels_list), np.array(predictions_list), target_names=["O", "B-Chemical", "I-Chemical"]))

    
    def _read_results_file(self):
        preds, labels = [], []
        with open(self.results_file, 'r') as file_:
            for line in file_.readlines():
                pred, label = line.split(self.separator)
                preds.append(pred)
                labels.append(label)
            assert len(preds) == len(labels), """Length of predictions and labels must be the same"""
        return preds, labels

    def _string_to_list(self,prediction):
        return prediction.split()

    def _from_text_to_predictions(self,text):
        
        prediction = []
        for word in text:
            match = re.match(self.tagged_entities_regex, word)
            if match:
                tag = self.label2id.get(match.string.split(":")[0], 0)
                prediction.append(tag)
            else:
                prediction.append(0)
        return prediction

    def _get_single_result_full_text(self, p, l):
        label_list = l.split()
        total_labels_in_example = len(label_list)
        predictions = self._from_text_to_predictions(p.split())
        expected = self._from_text_to_predictions(l.split())
        if len(predictions) == len(expected):
            return predictions, expected
        else:
            if total_labels_in_example > 1:
                alt_predictions = []
                alt_expected = []
                predictions_label_name_pair = re.findall(self.tagged_entities_regex, p)
                expected_label_name_pair = re.findall(self.tagged_entities_regex, l)
                for pair in expected_label_name_pair:
                    alt_expected.append(self.label2id[pair[0]])
                    if pair in predictions_label_name_pair:
                        alt_predictions.append(self.label2id[pair[0]])
                        predictions_label_name_pair.remove(pair)
                    if pair not in predictions_label_name_pair:
                        alt_predictions.append(0)
                for pair in predictions_label_name_pair:
                    alt_predictions.append(self.label2id.get(pair[0],0))
                while len(alt_predictions) < total_labels_in_example:
                    alt_predictions.append(0)
                while len(alt_expected) < total_labels_in_example:
                    alt_expected.append(0)
                assert len(alt_predictions) == len(alt_expected)  , f"Not Same lengths {len(alt_predictions)} {len(alt_expected)}"
                return alt_predictions, alt_expected
            else:
                return [], []



