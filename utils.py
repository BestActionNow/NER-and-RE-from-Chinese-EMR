import torch
import os
import pickle

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# Construct vocabulary from setences
def build_vocab(params):
    lines = []
    for path in params.all_data:
        with open(os.path.join(params.data_path, path)) as f:
            lines_ = f.readlines()
        lines.extend(lines_)
    word2idx={}
    for line in lines:
        for word in line:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

def decode_tags(sent, ixs, ix_to_tag):
    tags = []
    for ix in ixs:
        tags.append(ix_to_tag[ix])

    # correct the annotated tags for the sentence
    for i, tag in enumerate(tags):
        if tag[0] == "I" and not (tags[i - 1] == str("B-" + tag[-1]) or tags[i - 1] == tag):
            tags[i] = "O"
    
    # output the annotated sentence
    i = 0
    output = []
    while i < len(tags):
        if tags[i][0] == "B":
            output.append(" ")
            end = i + 1
            while end < len(tags) and tags[end] == "I" + tags[i][1: ]:
                end += 1
            output.extend(sent[i: end]) 
            output.append("\\" + tags[i][-1].lower() + " ")
            i = end
        else:
            output.extend(sent[i])
            i += 1
    return "".join(output)


    
