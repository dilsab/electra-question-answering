import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AdamW

from dataset import Dataset
from utils import load_model_tokenizer, last_save_path


def get_args():
    parser = argparse.ArgumentParser('ELECTRA Question answering - dilsab')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_data', type=str, default='data/train.json', help='Train data path')
    parser.add_argument('--validation_data', type=str, default='data/validation.json', help='Validation data path')
    parser.add_argument('--validation_interval', type=int, default=1, help='Number of epochs between validation phases')
    parser.add_argument('--load_last', action='store_true', help='Load last model and tokenizer')
    parser.add_argument('--save_path', type=str, default='save', help='Model and tokenizer save path')
    parser.add_argument('--save_interval', type=int, default=300,
                        help='Save model and tokenizer every save_interval steps')

    return parser.parse_args()


def add_answer_end_indexes(answers):
    for answer in answers:
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])


def save(model, tokenizer, save_path, epoch, step):
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(save_path, f'electra_{epoch}_{step}')
    if os.path.isdir(save_dir):
        return
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Saved {save_dir}')


def read_data(filename):
    contexts = []
    questions = []
    answers = []
    with open(Path(filename), 'rb') as f:
        data_dict = json.load(f)

    for data in data_dict['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_start_end_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_position = encodings.char_to_token(i, answers[i]['answer_start'])
        end_position = encodings.char_to_token(i, answers[i]['answer_end'] - 1)

        if start_position is None:
            start_position = tokenizer.model_max_length
        if end_position is None:
            end_position = tokenizer.model_max_length

        start_positions.append(start_position)
        end_positions.append(end_position)

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def train(options):
    print('ELECTRA transformer training.')

    train_contexts, train_questions, train_answers = read_data(options.train_data)
    validation_contexts, validation_questions, validation_answers = read_data(options.validation_data)

    add_answer_end_indexes(train_answers)
    add_answer_end_indexes(validation_answers)

    total_steps = 0
    total_epochs = 0

    # Load model
    model_tokenizer_path = 'google/electra-small-discriminator'
    if options.load_last:
        model_tokenizer_path = last_save_path(options.save_path)
        filename_parts = os.path.basename(model_tokenizer_path).split('_')
        total_steps = int(filename_parts[-1])
        total_epochs = int(filename_parts[-2])

    model, tokenizer = load_model_tokenizer(model_tokenizer_path)
    print(f'Loaded model and tokenizer from {model_tokenizer_path}.')
    print(f'Step: {total_steps}')
    print(f'Epoch: {total_epochs}')

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    validation_encodings = tokenizer(validation_contexts, validation_questions, truncation=True, padding=True)

    add_start_end_token_positions(train_encodings, train_answers, tokenizer)
    add_start_end_token_positions(validation_encodings, validation_answers, tokenizer)

    train_dataset = Dataset(train_encodings)
    validation_dataset = Dataset(validation_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=options.batch_size)

    optimizer = AdamW(model.parameters(), lr=options.learning_rate)

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(validation_dataset)}')

    train_batches = len(train_loader)
    validation_batches = len(validation_loader)
    print(f'Train data batches: {train_batches}')
    print(f'Validation data batches: {validation_batches}')

    step = 0
    for epoch in range(options.epochs):
        # Training
        losses = []
        tqdm_progress = tqdm(train_loader)
        for i, data in enumerate(tqdm_progress):
            tqdm_progress.update()
            optimizer.zero_grad()
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            start_positions = data['start_positions'].to(device)
            end_positions = data['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            losses.append(float(loss))
            optimizer.step()
            step += 1
            total_steps += 1
            tqdm_progress.set_description(
                '[Train] Epoch: {}/{}, total epochs: {}, batch: {}/{}, step: {}, total steps: {}. Loss: {:.7f}'.format(
                    epoch, options.epochs, total_epochs, i + 1, train_batches, step, total_steps, float(loss)))
            if step % options.save_interval == 0:
                save(model, tokenizer, options.save_path, total_epochs, total_steps)
        print(f'[Train] Epoch: {epoch}/{options.epochs}, total epochs: {total_epochs}. Mean loss: {np.mean(losses)}')

        # Save after every epoch
        total_epochs += 1
        save(model, tokenizer, options.save_path, total_epochs, total_steps)
        if (epoch + 1) % options.validation_interval == 0:
            # Validation
            losses = []
            model.eval()
            tqdm_progress = tqdm(validation_loader)
            for i, data in enumerate(tqdm_progress):
                tqdm_progress.update()
                with torch.no_grad():
                    input_ids = data['input_ids'].to(device)
                    attention_mask = data['attention_mask'].to(device)
                    start_positions = data['start_positions'].to(device)
                    end_positions = data['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    losses.append(float(loss))
                    tqdm_progress.set_description(
                        '[Validation] Batch: {}/{}. Loss: {:.7f}'.format(
                            i + 1, validation_batches, float(loss)))
            print('[Validation] Epoch: {}/{}, total epochs: {}. Mean loss: {}'.format(
                epoch + 1, options.epochs, total_epochs, np.mean(losses)
            ))

            model.train()


if __name__ == '__main__':
    train(get_args())
