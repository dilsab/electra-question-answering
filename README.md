# electra-question-answering

ELECTRA transformer training using [Hugging Face transformers](https://huggingface.co/transformers/index.html).

## Installation

**CPU**
```shell
pipenv install
```
Or
```shell
pip install transformers[torch], numpy, tqdm
```
**GPU**\
Install [PyTorch](https://pytorch.org/). \
Run
```shell
pipenv install transformers, numpy, tqdm
```

## Usage

### Command line arguments (optional)

| Argument              | Type              | Default               | Description |
| --------------------- | ----------------- | --------------------- | ----------- |
| --learning_rate       | float             | 5e-4                  | Learning rate |
| --epochs              | int               | 10                    | Number of epochs |
| --batch_size          | int               | 32                    | Batch size |
| --train_data          | str               | data/train.json       | Train data path |
| --validation_data     | str               | data/validation.json  | Validation data path |
| --validation_interval | int               | 1                     | Number of epochs between validation phases |
| --load_last           | No value argument | -                     | Load last model and tokenizer |
| --save_path           | str               | save                  | Model and tokenizer save path |
| --save_interval       | int               | 300                   | Save model and tokenizer every save_interval steps |

Activate Pipenv shell
```shell
pipenv shell
```
Run command
```shell
python train.py <arguments>
