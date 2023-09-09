import pickle
import random
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import openpyxl
from sklearn.model_selection import train_test_split


# supersenses acknowleged
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']


def get_training_set(seeds_file_path="train_data.txt"):
    with open(seeds_file_path, 'r', encoding="utf-8") as seeds_file:
        train = []
        # build dictionnary structures
        headers = seeds_file.readline().strip("\n").strip().split("\t")
        while 1:
            line = seeds_file.readline()
            if not line:
                break
            sline = line.strip("\n").strip().split("\t")
            examples = []
            for header, value in zip(headers, sline):
                if header == "supersense":
                    supersense = value
                if header == "lemma":
                    lemma = value
                if header == "labels":
                    labels = value
                if header == "definition":

                    definition = value

                    definition_with_lemma = f"{lemma} : {value}"

                    if labels:
                        definition_with_labels = ""
                        definition_with_labels += f"({labels}) "
                        definition_with_labels += value

                        definition_with_lemma_and_labels = f"{lemma} : "
                        definition_with_lemma_and_labels += f"({labels}) "
                        definition_with_lemma_and_labels += value
                    else:
                        definition_with_labels = ""
                        definition_with_labels += value

                        definition_with_lemma_and_labels = f"{lemma} : "
                        definition_with_lemma_and_labels += value

                if "example" in header:
                    if value:
                        examples.append(value)

            if supersense in SUPERSENSES:
                train_dic = {}
                train_dic["definition"] = definition
                train_dic["definition_with_lemma"] = definition_with_lemma
                train_dic["definition_with_labels"] = definition_with_labels
                train_dic["definition_with_lemma_and_labels"] = definition_with_lemma_and_labels
                train_dic["examples"] = examples
                train_dic["supersense"] = supersense
                train.append(train_dic)

                print(definition)
                print(definition_with_lemma)
                print(definition_with_labels)
                print(definition_with_lemma_and_labels)
                print(supersense)
                print("")

        random.shuffle(train)

        # serialize the different structures created
        with open(f"train.pkl", "wb") as file:
            pickle.dump(train, file)


def get_dev_test_sets(eval_file_path="eval_data.txt"):

    with open(eval_file_path, 'r', encoding="utf-8") as eval_file:
        dev_test = []
        # build dictionnary structures
        headers = eval_file.readline().strip("\n").strip().split("\t")
        while 1:
            line = eval_file.readline()
            if not line:
                break
            sline = line.strip("\n").strip().split("\t")
            examples = []
            for header, value in zip(headers, sline):
                if header == "supersense":
                    supersense = value
                if header == "lemma":
                    lemma = value
                if header == "labels":
                    labels = value
                if header == "definition":
                    definition = value

                    definition_with_lemma = f"{lemma} : {value}"

                    if labels:
                        definition_with_labels = ""
                        definition_with_labels += f"({labels}) "
                        definition_with_labels += value

                        definition_with_lemma_and_labels = f"{lemma} : "
                        definition_with_lemma_and_labels += f"({labels}) "
                        definition_with_lemma_and_labels += value
                    else:
                        definition_with_labels = ""
                        definition_with_labels += value

                        definition_with_lemma_and_labels = f"{lemma} : "
                        definition_with_lemma_and_labels += value

                if "example" in header:
                    if value:
                        examples.append(value)

            if supersense in SUPERSENSES:
                dev_test_dic = {}
                dev_test_dic["definition"] = definition
                dev_test_dic["definition_with_lemma"] = definition_with_lemma
                dev_test_dic["definition_with_labels"] = definition_with_labels
                dev_test_dic["definition_with_lemma_and_labels"] = definition_with_lemma_and_labels
                dev_test_dic["examples"] = examples
                dev_test_dic["supersense"] = supersense
                dev_test.append(dev_test_dic)

                print(definition)
                print(definition_with_lemma)
                print(definition_with_labels)
                print(definition_with_lemma_and_labels)
                print(supersense)
                print("")

        # build id lists for classification sets
        dev, test = train_test_split(dev_test, test_size=0.5, random_state=42)

        # serialize the different structures created
        with open(f"dev.pkl", "wb") as file:
            pickle.dump(dev, file)
        with open(f"test.pkl", "wb") as file:
            pickle.dump(test, file)


def statistical_analysis(train_data, dev_test_data, wiki):
    pass

"""
get_training_set()
get_dev_test_sets()
"""

"""
train_dist = {supersense : 0 for supersense in SUPERSENSES}
dev_dist = {supersense : 0 for supersense in SUPERSENSES}

with open(f"train.pkl", "rb") as file:
    train = pickle.load(file)
with open(f"dev.pkl", "rb") as file:
    dev = pickle.load(file)

for example in train:
    train_dist[example["supersense"]] += 1

for example in dev:
    dev_dist[example["supersense"]] += 1

print(f"NB_EXAMPLES_SEEDS = {len(train)}")
print(f"NB_EXAMPLES_DEV = {len(dev)}\n")

for supersense in SUPERSENSES:
    print(f"NB_EXAMPLES_FOR_{supersense}_IN_SEEDS = {train_dist[supersense]}")
    print(f"NB_EXAMPLES_FOR_{supersense}_IN_DEV = {dev_dist[supersense]}\n")



class_names = list(train_dist.keys())
class_ids = [string.ascii_lowercase[i % 26] for i in range(len(class_names))]
values_train = list(train_dist.values())
values_dev = list(dev_dist.values())

width = 0.35
x = range(len(class_names))
fig, ax = plt.subplots()
rects1 = ax.bar(x, values_train, width, label='train')
rects2 = ax.bar([i + width for i in x], values_dev, width, label='dev')

ax.set_xlabel('class')
ax.set_ylabel('nb examples')
ax.set_title('nb examples for each class')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(class_ids)
ax.legend()

plt.savefig('data_dist.png')

for id, classe in zip(class_ids, class_names):
    print(f"{id} : {classe}")
"""