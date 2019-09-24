# Author: Xu Zhao

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle
from utils import *


torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, params):
        super(BiLSTM_CRF, self).__init__()
        self.device = "cuda:0" if params.cuda else "cpu" 
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(self.device)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.,device=self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def train_NER_model(model ,train_data, tag_to_ix, word_to_ix, epoch_num, device):

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        for index, (sentence, tags) in enumerate(train_data):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).cuda()

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            if index % 20 == 0:
                print("iteration {}/{} completed, loss: {:.4f}".format(index, len(train_data), loss.item()))
        print("epoch {}/{} completed, loss: {:.4f}".format(epoch, epoch_num, loss.item()))
    
    return model

def test_NER_model(model, test_data, tag_to_ix, word_to_ix):

    ix_to_tag = {} 
    for key in tag_to_ix:
        ix_to_tag[tag_to_ix[key]] = key
    output = []
    for index, line in enumerate(test_data):
        with torch.no_grad():
            encoded_sent = prepare_sequence(line.strip("\n"), word_to_ix).cuda()
            tag_ixs = model(encoded_sent)[1]
            annotated_line = decode_tags(line.strip("\n"), tag_ixs, ix_to_tag) 
        output.append(annotated_line.strip() + "\n")
        if index % 20 == 0:
            print("{}/{} items complete testing".format(index, len(test_data)))
    return output


if __name__ == '__main__':

    with open("./data/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
             
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

    word_to_ix = build_vocab("./data/raw_data.orig")

    device = "cuda:0"

    save_path = "./model.pkl"

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, device).cuda()
    train_NER_model(model, train_data, tag_to_ix, word_to_ix, device)
    with open(save_path, "wb") as f:
        torch.save(model, f)
    # model = torch.load("./model.pkl")
    ix_to_tag = {} 
    for key in tag_to_ix:
        ix_to_tag[tag_to_ix[key]] = key

    with torch.no_grad():
        sent = prepare_sequence(train_data[0][0], word_to_ix).cuda()
        score, tags = model(sent)
        print(decode_tags(sent, tags, ix_to_tag))