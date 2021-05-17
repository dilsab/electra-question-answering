import os
from glob import glob

from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering


def load_model_tokenizer(path):
    return ElectraForQuestionAnswering.from_pretrained(path), \
           ElectraTokenizerFast.from_pretrained(path)


def last_save_path(save_path):
    save_paths = glob(os.path.join(save_path, '*'))
    last_path = sorted(save_paths, key=lambda x: int(x.rsplit('_')[-1]), reverse=True)[0]

    return last_path
