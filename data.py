import pickle
import random
from collections import defaultdict

import openpyxl
from sklearn.model_selection import train_test_split


# supersenses acknowleged
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']


def get_training_set(seeds_file_path="new_seeds_V2.txt", wiki_file="wiki_dump.pkl"):
    with open(wiki_file, "rb") as file:
        wiki = pickle.load(file)
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
            labels = []
            for header, value in zip(headers, sline):
                if header == "id_sense_wiki":
                    sense_id = value
                    if wiki.lexical_senses[sense_id].labels:
                        labels = list(wiki.lexical_senses[sense_id].labels)
                if header == "supersense":
                    supersense = value
                if header == "lemma":
                    lemma = value
                if header == "definition":
                    value = value.replace("DEF:", "")
                    definition = value
                    definition_with_lemma = f"{lemma} : {value}"
                    definition_with_labels = ""
                    if labels:
                        for label in labels:
                            definition_with_labels += f"({label}) "
                        definition_with_labels += value
                    else:
                        definition_with_labels = value
                    definition_with_lemma_and_labels = f"{lemma} : "
                    if labels:
                        for label in labels:
                            definition_with_lemma_and_labels += f"({label}) "
                        definition_with_lemma_and_labels += value
                    else:
                        definition_with_lemma_and_labels = f"{lemma} : {value}"
                if "example" in header:
                    if value:
                        value = value.replace("EX:", "")
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
        random.shuffle(train)
        # serialize the different structures created
        with open(f"train.pkl", "wb") as file:
            pickle.dump(train, file)


def get_dev_test_sets(seeds_file_path="dev_test_data.txt", wiki_file="wiki_dump.pkl"):
    with open(wiki_file, "rb") as file:
        wiki = pickle.load(file)

    with open(seeds_file_path, 'r', encoding="utf-8") as seeds_file:
        dev_test = []
        # build dictionnary structures
        headers = seeds_file.readline().strip("\n").strip().split("\t")
        while 1:
            line = seeds_file.readline()
            if not line:
                break
            sline = line.strip("\n").strip().split("\t")
            examples = []
            labels = []
            for header, value in zip(headers, sline):
                if header == "id_sense_wiki":
                    sense_id = value
                    if wiki.lexical_senses[sense_id].labels:
                        labels = list(wiki.lexical_senses[sense_id].labels)
                if header == "supersense":
                    supersense = value
                if header == "lemma":
                    lemma = value
                if header == "definition":
                    value = value.replace("DEF:", "")
                    definition = value
                    definition_with_lemma = f"{lemma} : {value}"
                    definition_with_labels = ""
                    if labels:
                        for label in labels:
                            definition_with_labels += f"({label}) "
                        definition_with_labels += value
                    else:
                        definition_with_labels = value
                    definition_with_lemma_and_labels = f"{lemma} : "
                    if labels:
                        for label in labels:
                            definition_with_lemma_and_labels += f"({label}) "
                        definition_with_lemma_and_labels += value
                    else:
                        definition_with_lemma_and_labels = f"{lemma} : {value}"
                if "example" in header:
                    if value:
                        value = value.replace("EX:", "")
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

        # build id lists for classification sets
        dev, test = train_test_split(dev_test, test_size=0.5, random_state=42)

        # serialize the different structures created
        with open(f"dev.pkl", "wb") as file:
            pickle.dump(dev, file)
        with open(f"test.pkl", "wb") as file:
            pickle.dump(test, file)


def logs_reader(logs_file):
    pass


def statistical_analysis(seeds, wiki):
    pass

