import argparse
import torch
import wiktionary as wi
import classifier as clf


# supersenses acknowleged
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon", "act*cognition"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person", "groupxperson"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance", "artifact*cognition"],
               "informational_object": ["cognition", "communication", "act*cognition", "artifact*cognition"],
               "quantification": ["quantity", "part", "group", "groupxperson"],
               "other": ["institution", "possesion", "time"]
               }

# relations to be acknowledged while reading wiktionary ttl files
allowed_relations = ["rdf:type",
                     "lexinfo:partOfSpeech",
                     "ontolex:canonicalForm",
                     "ontolex:sense",
                     "dbnary:describes",
                     "dbnary:synonym",
                     "lexinfo:gender",
                     "skos:definition",
                     "skos:example",
                     "rdfs:label",
                     "dbnary:partOfSpeech"]

# rdf types to be acknowledged while reading wiktionary ttl files
allowed_rdf_types = ["ontolex:LexicalSense",
                     "ontolex:Form",
                     "ontolex:LexicalEntry",
                     "dbnary:Page",
                     "ontolex:Word",
                     "ontolex:MultiWordExpression"]

# grammatical categories to be acknowledged while reading wiktionary ttl files
allowed_categories = ["lexinfo:noun", '"-nom-"']

# labels in definitions leading to ignore a lexical sense while reading wiktionary ttl files
labels_to_ignore = ["vieilli", "archaïque", "désuet", "archaïque, orthographe d’avant 1835"]

# language token to be acknowledged while reading wiktionary ttl files
lang = "fra"


class Parameters:
    def __init__(self, nb_epochs=100, batch_size=25, hidden_layer_size=300, patience=5, lr=0.00025, frozen=True, max_seq_length=50, window_example=10, definition_mode='definition'):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.patience = patience
        self.lr = lr
        self.frozen = frozen
        self.max_seq_length = max_seq_length
        self.window_example = window_example
        self.definition_mode = definition_mode
        self.keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "max_seq_length", "window_example", "definition_mode"]


# parameter combinations to find the values to make the best classifier
PARAMETER_COMBINATIONS = []
# best parameters to serialize
BEST_PARAMETERS = Parameters()
# basic parameters to test the classifier
TEST_PARAMETERS = Parameters()


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_mode", choices=['parse', 'classify'], help="Sets the main purpose of the program, that is to say parsing a file linked to seeds or use a classifier for supersense classification.")
    parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
    parser.add_argument('-parsing_file', default="fr_dbnary_ontolex.ttl", help='wiktionary file (dbnary dump, in turtle format) or serialized Wiktionary instance (pickle file).')
    parser.add_argument("-checked_seeds_file", default="train_data.txt", help="")
    parser.add_argument("-train_file", default="train.pkl", help="")
    parser.add_argument("-dev_file", default="dev.pkl", help="")
    parser.add_argument("-test_file", default="test.pkl", help="")
    parser.add_argument("-definition_mode", choices=['definition', 'definition_with_lemma','definition_with_labels', 'definition_with_lemma_and_labels'], default="definition", help="")
    parser.add_argument("-corpus_file", default="sequoia.deep_and_surf.parseme.frsemcor", help="")
    parser.add_argument('-parsing_mode', choices=['read', 'filter', 'read_and_dump'], help="Sets the mode for the parsing: read, filter or read_and_dump.")
    parser.add_argument("-inference_data_file", default=None, help="File containing the data for inference.")
    parser.add_argument("-wiktionary_dump", default="wiki_dump.pkl", help="Serialized Wiktionary instance containig all the annoted data for the classifier to be trained and evaluated.")
    parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser_args()

    if args.main_mode == "parse":
        parsing_file = args.parsing_file

        # creation of an instance of the Parser class meant to parse wiktionary related files
        wiki_parser = wi.Parser(categories=allowed_categories,
                                relations=allowed_relations,
                                lang=lang,
                                labels_to_ignore=labels_to_ignore,
                                trace=args.trace,
                                wiki_dump=args.wiktionary_dump,
                                rdf_types=allowed_rdf_types,
                                file=parsing_file
                                )
        wiki_parser.set_parser_mode(args.parsing_mode)
        wiki_parser.parse_file()

    if args.main_mode == "classify":

        # DEVICE setup
        device_id = args.device_id
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda:" + args.device_id)

        def_mode = args.definition_mode

        # Classification program
        with open("logs_file.txt", 'w', encoding="utf-8") as file:
            for i in range(5):
                train_examples, dev_examples, test_examples = clf.encoded_examples_split(def_mode,
                                                                                         train=args.train_file,
                                                                                         dev=args.dev_file,
                                                                                         test=args.test_file)
                # for lr in [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
                for lr in [0.00005, 0.00003, 0.00002, 0.00001, 0.000005, 0.000001]:
                    for patience in [5]:

                        print("")
                        print(f"run {i+1} : lr = {lr}; mode = {def_mode}")
                        print("")

                        hypersense_dist = {hypersense: 0 for hypersense in HYPERSENSES}
                        hypersense_correct = {hypersense: 0 for hypersense in HYPERSENSES}
                        supersense_dist = {supersense: 0 for supersense in SUPERSENSES}
                        supersense_correct = {supersense: 0 for supersense in SUPERSENSES}

                        params = Parameters(lr=lr, definition_mode=def_mode, patience=patience, frozen=False)

                        file.write(f"run:{i + 1};")

                        classifier = clf.SupersenseTagger(params, DEVICE)
                        clf.training(params, train_examples, dev_examples, classifier, DEVICE, file)
                        clf.evaluation(dev_examples, classifier, DEVICE, file, supersense_dist,
                                       supersense_correct, hypersense_dist, hypersense_correct)

                        sequoia_baseline = clf.MostFrequentSequoia(args.corpus_file)
                        train_baseline = clf.MostFrequentTrainingData(args.train_file)
                        wiki_baseline = clf.MostFrequentWiktionary(args.wiktionary_dump)

                        sequoia_baseline.training()
                        train_baseline.training()
                        wiki_baseline.training()

                        file.write(f"sequoia_baseline:{sequoia_baseline.evaluation(args.dev_file)};")
                        file.write(f"train_baseline:{train_baseline.evaluation(args.dev_file)};")
                        file.write(f"wiki_baseline:{wiki_baseline.evaluation(args.dev_file)};")

                        file.write("\n")
