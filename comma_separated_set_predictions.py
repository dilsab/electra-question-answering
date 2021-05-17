import argparse
import csv
from pathlib import Path

from tqdm.autonotebook import tqdm

from answering import ElectraQuestionAnswering


def get_args():
    parser = argparse.ArgumentParser('Comma-separated contexts, questions predictions - dilsab')
    parser.add_argument('--data_path', type=str, default='data/comma_separated_test_set.csv', help='Path to data file')
    parser.add_argument('--out_file_path', type=str, default='out/comma_separated_predictions.csv',
                        help='Predictions file save path')

    return parser.parse_args()


def read_data(filename):
    contexts = []
    questions = []
    with open(Path(filename), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            context, question = row
            contexts.append(context)
            questions.append(question)

    return contexts, questions


def predict(options):
    electra_question_answering = ElectraQuestionAnswering('save')

    contexts, questions = read_data(options.data_path)

    predictions = []
    for context, question in tqdm(zip(contexts, questions), total=len(questions)):

        predictions.append(electra_question_answering.predict_answer(context, question))

    file_path = options.out_file_path
    with open(file_path, 'w') as file:
        for prediction in predictions:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow([prediction] if prediction else [])
        print(f'Created file {file_path}')


if __name__ == '__main__':
    predict(get_args())
