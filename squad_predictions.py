import argparse
import json
from pathlib import Path

from tqdm.autonotebook import tqdm

from answering import ElectraQuestionAnswering


def get_args():
    parser = argparse.ArgumentParser('SQuAD2.0 dataset predictions - dilsab')
    parser.add_argument('--data_path', type=str, default='data/dev-v2.0.json', help='Path to data file')
    parser.add_argument('--out_file_path', type=str, default='out/dev-v2.0_predictions.json',
                        help='Predictions file save path')

    return parser.parse_args()


def read_data(filename):
    contexts = []
    questions = []
    ids = []
    with open(Path(filename), 'rb') as f:
        data_dict = json.load(f)

    for data in data_dict['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                contexts.append(context)
                questions.append(qa['question'])
                ids.append(qa['id'])

    return contexts, questions, ids


def predict(options):
    electra_question_answering = ElectraQuestionAnswering('save')

    contexts, questions, ids = read_data(options.data_path)

    predictions = {}
    for context, question, id in tqdm(zip(contexts, questions, ids), total=len(ids)):
        predictions[id] = electra_question_answering.predict_answer(context, question)

    file_path = options.out_file_path
    with open(file_path, 'w') as file:
        json.dump(predictions, file)
        print(f'Created file {file_path}')


if __name__ == '__main__':
    predict(get_args())
