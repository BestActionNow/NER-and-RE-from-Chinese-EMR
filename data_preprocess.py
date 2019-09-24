import os
import pickle

def resolve_line(line_num, origin_line, line, entity_type_list):
    words = []
    tags = []
    for part in line.split(" "):
        part = part.strip("\n")
        if len(part) >= 2 and part[-2] == '\\' and part[-1] in entity_type_list:
            words.extend(list(part[ :-2]))
            part_tag = ["B-" + part[-1].upper()] + ["I-" + part[-1].upper()] * (len(part) - 3)
            tags.extend(part_tag)
        else:
            words.extend(list(part))
            part_tag = ["O"] * len(part)
            tags.extend(part_tag)
    assert len(words) == len(tags)
    if(list(origin_line.strip()) != words):
        print("At line {} origin and annotated line don't match! ".format(line_num))
    assert words == list(origin_line.strip())
    return words, tags

# convert the annotation file
def read_corpora(params, entity_type_list, save_path):
    data_origin = os.path.join(params.data_path, params.train_origin_data)
    annotation = os.path.join(params.data_path, params.train_annotated_data)
    with open(data_origin) as f1:
        origin_lines = f1.readlines()
    with open(annotation) as f2:
        annotated_lines = f2.readlines()
    assert len(origin_lines) == len(annotated_lines)
    train_data = []
    for i in range(len(origin_lines)):
        words, tags = resolve_line(i+1, origin_lines[i], annotated_lines[i], entity_type_list)
        train_data.append((words, tags))
    with open(save_path, "wb") as f:
        pickle.dump(train_data, f)
    return train_data

