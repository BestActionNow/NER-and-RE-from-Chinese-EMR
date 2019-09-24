import torch
import argparse
import os
from utils import decode_tags, prepare_sequence
from NER_model import *
from data_preprocess import read_corpora

## hyperparameters for NER
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--data_path', type=str, default='data', help='train data source')
parser.add_argument('--train_origin_data', type=str, default='data.orig', help='the origin data for training')
parser.add_argument('--train_annotated_data', type=str, default='annotation.man.af', help='the annotated data for training')
parser.add_argument('--test_data', type=str, default="raw_data.orig", help="data for testing")
parser.add_argument('--test_output', type=str, default="output", help="the annotated lines for test data")
parser.add_argument('--all_data', type=list, nargs = "+", default=["raw_data.orig"], help="all data including training and testing")
parser.add_argument('--cuda', type=bool, default=True, help='if use gpu for training or testing')
parser.add_argument('--embedding_dim', type=int, default=300, help='the word vector dim')
parser.add_argument('--hidden_dim', type=int, default=600, help='the hidden layer dim for bi-lstm')
parser.add_argument('--epoch_num', type=int, default=10, help='total epoch number for model training')
parser.add_argument('--init_model', type=str, default="", help='use a pre-trained model to continue training or testing')
params = parser.parse_args()


# tag to label index, need be setted
tag_to_ix = { "O": 0,
              "B-D": 1, "I-D": 2,
              "B-S": 3, "I-S": 4,
              "B-T": 5, "I-T": 6,
              "B-I": 7, "I-I": 8,
              "B-C": 9, "I-C": 10,
              "B-A": 11, "I-A": 12,
              "B-B": 13, "I-B": 14,
              "B-P": 15, "I-P": 16,
              "<START>": 17, "<STOP>": 18
            }
entity_type_list = ["d", "s", "c", "i", "a", "b", "t", "p"]

# build vocab according to all of the data
word_to_ix = build_vocab(params)

# init model for training
if params.init_model != str(""):
    model = torch.load(os.path.join(params.data_path, params.init_model))
else:
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, params)
if params.cuda:
    model = model.cuda() 
    device = "cuda:0"
else:
    device = "cpu" 

# set training data
if os.path.exists(os.path.join(params.data_path, "train_data.pkl")):
    with open(os.path.join(params.data_path, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
else:
    train_data = read_corpora(params, entity_type_list, os.path.join(params.data_path, "train_data.pkl"))

# set testing data
if os.path.exists(os.path.join(params.data_path, params.test_data)):
    with open(os.path.join(params.data_path, params.test_data), "r") as f:
        test_data = f.readlines()

# train and save model 
model = train_NER_model(model, train_data, tag_to_ix, word_to_ix, params.epoch_num, device)
with open(os.path.join(params.data_path, "model.pkl"), "wb") as f:
    torch.save(model, f)

# test model
with torch.no_grad():
    annotated_lines = test_NER_model(model, test_data, tag_to_ix, word_to_ix)
with open(os.path.join(params.data_path, params.test_output), "w") as f:
    for line in annotated_lines:
        f.write(line)