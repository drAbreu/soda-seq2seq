import datasets
from typing import Tuple, List
TASKS = {
  "BC5CDR-chem-IOB":"NER chemical mentions",
  "BC5CDR-disease-IOB":"NER disease mentions",
  "BC2GM-IOB": "NER gene mentions",
  "NCBI-disease-IOB":"NER disease mentions",
  "JNLPBA":"NER jnlpba protein, cell_type, cell_line, DNA, RNA mentions",
  "BIOSSES":"Sentence similarity",
  "PubMedQA": "Question answering"
}

class BlurbIOBToSeq2Seq:
    """Converts the BLURB datasets, or any dataset follwoing the
    IOB tagging schema into a sequence to sequence problem. Any 
    dataset out of BLURB can be added, given that a task for it is added
    to the TASK dictionary and the dataset follows the:
    {'tokens': List['str'], 'ner_tags': List[int]}.
    And the dataset is loadable using `datasets.load_dataset`.
    """
    def __init__(self,
                 dataset_name: str = "BC5CDR-chem-IOB",
                 separator: str = "|separator|",
                 label_mode: str = "full-text",
                 output: str = None, #gpt or hf,
                 closing_token_outputs: str = "[END]", # Needed for GPT-3 API
                 closing_token_inputs: str = "\n\n[END]\n\n", # Needed for GPT-3 API
                 class_start: str = "_class_***", # Needed for GPT-3 API
                 class_end: str = "***_class_" # Needed for GPT-3 API
                 ):
        """Converts the BLURB datasets, or any dataset follwoing the
            IOB tagging schema into a sequence to sequence problem.

        Args:
            dataset_name (str, optional): Name of the dataset in the EMBO/BLURB dataset repository. Defaults to "BC5CDR-chem-IOB".
            separator (str, optional): string to separate to consecutive tag mentions in the generated text.
                                        Ignored if `label_mode` is not `labels-only`.  Defaults to "|separator|".
            label_mode (str, optional): Possible values are `labels-only` or `full-text`. Modifies the output string
                                        of the dataset. `labels-only` lists the labels mentioned in the input. `
                                        full-text` repeats the entire text, adding the labels in the text. The 
                                        labels are indicated with `class_start` and `class_end`. Relevant only f
                                        or token clasification (NER) tasks.. Defaults to "full_text".
            output (str, optional): Whether to generate the dataset on a `gpt` or 🤗 (`hf`) format. Defaults to None.
            closing_token_outputs (str, optional): Indicates end of sentence in outputs. To be added to inference in GPT. Defaults to "[END]".
            closing_token_inputs (str, optional): Indicates end of inputs. Defaults to "[END]".
            class_start (str, optional): Beginning of an entity in the text. Defaults to "[END]".
            class_end (str, optional): Indicates endd of entityu in text. Defaults to "[END]".
        """
        self.base_data = "EMBO/BLURB"
        self.dataset_name = dataset_name
        self.task_name = TASKS[self.dataset_name]
        self.model = model
        self.separator = separator
        self.label_mode = label_mode
        self.output = output
        self.closing_token_outputs = closing_token_outputs
        self.class_start = class_start
        self.class_end = class_end
        self.closing_token_inputs = closing_token_inputs
        self.dataset = datasets.load_dataset(self.base_data, self.dataset_name, download_mode='force_redownload')
        self.id2label, self.label2id = self._get_data_labels()

        self.dataset_seq2seq = self.dataset.map(
                                        self._convert_into_seq2seq,
                                        batched=True,
                                        remove_columns=["tokens", "ner_tags", "id"]
                                        )

    def _convert_into_seq2seq(self, examples) -> dict:
        """Converts IOB tagged datasets into Seq2seq datasets

        Args:
            examples (_type_): Batch of data generated by HuggingFace

        Raises:
            NotImplemented: Currently only supports NER related tasks

        Returns: batch of dict
            dict: {'prompt': str, 'completion': str} if `output='gpt'`
                  {'tokens': str, 'ner_tags': str} if `output='hf'`  
        """
        if "NER" in self.task_name:
            modified_data = self._ner_into_seq2seq(examples)
        else:
            raise NotImplemented
        return modified_data

    def _ner_into_seq2seq(self, examples) -> dict:
        inputs_ = 'prompt' if self.output == "gpt" else 'inputs'
        targets_ = 'completion' if self.output == "gpt" else 'targets'
        modified_data = {inputs_: [], targets_: []}

        for i, t in zip(examples['tokens'], examples['ner_tags']):
            modified_data[inputs_].append(f"""{self.task_name}: {" ".join(i)}{self.closing_token_inputs}""")
            modified_data[targets_].append(self._get_text_labels(i, t))

        return modified_data

    def _get_data_labels(self) -> Tuple[dict, dict]:
        """_summary_

        Returns:
            Tuple[dict, dict]: _description_
        """
        num_labels = self.dataset['train'].info.features['ner_tags'].feature.num_classes
        label_list = self.dataset['train'].info.features['ner_tags'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        print(f"\nTraining on {num_labels} features:")
        print(", ".join(label_list))
        return id2label, label2id

    def _get_text_labels(self, inputs: List[str], targets: List[int]) -> str:
        
        if self.label_mode == "labels-only":
            output_string = "We found the following chemical mentions -> "
            for i, t in zip(inputs, targets):
                if t != 0:
                    output_string += f"{self.id2label[t]}:{i} {self.separator}"
            if output_string == "We found the following chemical mentions -> ":
                output_string = "No entities found"
            output_string += f" {self.closing_token_outputs}"

        if self.label_mode == "full-text":
            inside_label = None
            output_string = ""
            for i, t in zip(inputs, targets):
                if inside_label:
                    if t == 0:
                        output_string += f""" {self.class_end.replace('_class_', inside_label)} {i}"""
                        inside_label = None
                    if self.id2label[t].startswith("B"):
                        output_string += f""" {self.class_end.replace("_class_", inside_label)} {self.class_start.replace("_class_", inside_label)} {i}"""
                    if self.id2label[t].startswith("I"):
                        output_string += f" {i}"
                else:
                    if t == 0:
                        output_string += f"{i} "
                    else:
                        if self.id2label[t].startswith("B"):
                            inside_label = self.id2label[t].split("-")[-1]
                            output_string += f""" {self.class_start.replace("_class_", inside_label)} {i}"""

            output_string += f" {self.closing_token_outputs}"
        return output_string

    def to_jsonl(self, path_to_file: str, splits: List[str]) -> None:
        """Writes the seq2seq dataset into a jsonl file. Specially useful to generate compatible GPT API data.

        Args:
            path_to_file (str): _description_
            splits (List[str]): _description_
        """
        for split in splits:
            self.dataset_seq2seq[split].to_json(f"{path_to_file}_{split}.jsonl", lines=True, orient='records')