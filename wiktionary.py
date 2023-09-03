import pickle
import random
import string
from collections import defaultdict
import openpyxl
from openpyxl.styles import PatternFill


def normalization_id(full_id, lang):
    if full_id.startswith(lang + ":"):
        return full_id.removeprefix(lang + ":").strip('_')
    elif full_id.startswith("<http://kaiko.getalp.org/dbnary/"):
        return full_id.removeprefix("<http://kaiko.getalp.org/dbnary/" + lang + "/").removesuffix(">").strip('_')
    else:
        return full_id


def extract_labels(text):
    if text:
        labels = set()
        label = ""
        definition = text
        labels_to_be_extracted = False
        if text.strip().startswith("("):
            labels_to_be_extracted = True
        while labels_to_be_extracted:
            end = definition.find(')')
            label = definition[:end + 1]
            labels.add(label[1:-1].lower())
            definition = definition.removeprefix(label).strip()
            if not definition.strip().startswith('('):
                labels_to_be_extracted = False
        return labels, definition
    else:
        return None, None


def extract_quote(left_delimiter='"', right_delimiter='"', text=""):
    quote = text
    stext = text.split("@fr")
    for el in stext:
        if "rdf:value" in el:
            el = el.replace("'", "’")
            start = el.find('"')
            end = el.rfind('"')
            quote = el[start + 1:end].strip()
            break
    return quote


class Page:

    def __init__(self, id, entry_ids=[]):
        self.id = id
        self.entry_ids = entry_ids

    def is_multi_entries_page(self):
        return len(self.entry_ids) > 1

    def get_entry_ids(self):
        return self.entry_ids

    def __str__(self):
        return f"PAGE_{self.id}: {str(self.entry_ids)}"

    def is_empty(self):
        return len(self.entry_ids) == 0


class lexicalEntry:

    def __init__(self, id, pos="", lemma="", synonyms=set(), sense_ids=[], morpho=[], cf_ids=[], is_mwe=False):
        self.id = id
        self.sense_ids = sense_ids
        self.cf_ids = cf_ids
        self.pos = pos
        self.lemma = lemma
        self.synonyms = synonyms
        self.morpho = morpho
        self.is_mwe = is_mwe

    def is_multi_senses_entry(self):
        return len(self.sense_ids) > 1

    def is_of_grammatical_category(self, grammatical_cats):
        return self.pos[0] in grammatical_cats

    def __str__(self):
        return f"   ENTRY_{self.id}: {self.sense_ids}\n"

    def add_morpho(self, morpho):
        self.morpho = morpho

    def set_sense_ids(self, sense_ids):
        self.sense_ids = sense_ids

    def get_lemma(self):
        return self.lemma

    def remove_sense_id(self, sense_id):
        self.sense_ids.remove(sense_id)

    def get_sense_ids(self):
        return self.sense_ids

    def get_cf_ids(self):
        return self.cf_ids

    def is_lexical_entry_of(self, page_id):
        self.is_lexical_entry_of = page_id

    def is_empty(self):
        return len(self.sense_ids) == 0


class lexicalSense:

    def __init__(self, id, definition="", examples=[()], synonyms=set(), labels=set()):
        self.id = id
        self.definition = definition
        self.examples = examples
        self.synonyms = synonyms
        self.labels = labels

    def __str__(self):
        string = f"       SENSE_{self.id}: \nDEF:\n{self.definition}"
        if len(self.examples) == 0:
            string += "\n"
            return string
        else:
            string += "\nEXAMPLES:\n"
            for example in self.examples:
                string += str(example) + "\n"
        return string

    def add_example(self, example):
        self.examples.append(example)

    def add_definition(self, definition):
        self.definition = definition

    def add_synonym(self, synonym):
        self.synonyms.add(synonym)

    def add_label(self, label):
        self.labels.add(label)

    def get_definition(self):
        return self.definition

    def get_examples(self):
        return self.examples

    def is_lexical_sense_of(self, entry_id):
        self.is_lexical_sense_of = entry_id

    def is_from_lemma(self, page_id):
        self.is_from_lemma = page_id


class Wiktionary:

    # constructor
    def __init__(self):
        self.pages = {}
        self.lexical_entries = {}
        self.lexical_senses = {}
        self.canonical_forms = {}

    def is_empty(self):
        return len(self.pages) == 0

    def get_pages(self):
        return self.pages

    def get_lexical_entries(self):
        return self.lexical_entries

    def get_lexical_senses(self):
        return self.lexical_senses

    def get_canonical_forms(self):
        return self.canonical_forms

    def pretty_print(self):
        for page_id in self.pages:
            print(self.pages[page_id])
            for entry_id in self.pages[page_id].get_entry_ids():
                print(self.lexical_entries[entry_id])
                for sense_id in self.lexical_entries[entry_id].get_sense_ids():
                    print(self.lexical_senses[sense_id])

    def add_page(self, page_id, page):
        # print("ADDING PAGE %s" % page_id)
        self.pages[page_id] = page

    def add_lexical_entry(self, entry_id, entry):
        # print("ADDING LE %s" % entry_id)
        self.lexical_entries[entry_id] = entry

    def add_lexical_sense(self, sense_id, sense):
        # print("ADDING SENSE %s" % sense_id)
        self.lexical_senses[sense_id] = sense

    def add_canonical_form(self, cf_id, gender):
        # print("ADDING CF %s" % cf_id)
        self.canonical_forms[cf_id] = gender

    def filter_wiki(self, trace=False):
        """
        Objectif : filtrage a posteriori des pages sans aucune LE de la bonne catégorie
                                     et  des LS correspondant à une LE de mauvaise catégorie
        """
        if trace:
            print("BEFORE FILTERING")
            print("===================================")
            print(self.pages.keys())
            print(self.lexical_entries.keys())
            print(self.lexical_senses.keys())

            print("\n")
            for page_id in self.pages:
                print(page_id, self.pages[page_id].get_entry_ids())

            print("\n")

            for entry_id in self.lexical_entries:
                print((entry_id, self.lexical_entries[entry_id].get_sense_ids()))

            print("\n")

        new_lexical_senses = {}
        new_lexical_entries = {}

        for entry_id in self.lexical_entries:
            le = self.lexical_entries[entry_id]
            sense_ids = le.get_sense_ids()
            new_sense_ids = []

            if sense_ids == None:
                continue
            else:
                for sense_id in sense_ids:
                    if sense_id not in self.lexical_senses:
                        continue
                    else:
                        sense = self.lexical_senses[sense_id]
                        sense.is_lexical_sense_of(entry_id)
                        new_lexical_senses[sense_id] = sense
                        new_sense_ids.append(sense_id)
                new_lexical_entries[entry_id] = le
            le.set_sense_ids(new_sense_ids)
        self.lexical_senses = new_lexical_senses
        self.lexical_entries = new_lexical_entries

        new_pages = {}
        for page_id in self.pages:
            p = self.pages[page_id]
            new_entry_ids = []
            for entry_id in p.get_entry_ids():
                if entry_id in self.lexical_entries:
                    self.lexical_entries[entry_id].is_lexical_entry_of(page_id)
                    new_entry_ids.append(entry_id)

            if len(new_entry_ids) > 0:
                p.entry_ids = new_entry_ids
                new_pages[page_id] = p
        self.pages = new_pages

        if trace:
            print("\nNB PAGES with at least one nominal entry %d" % len(self.pages.keys()))
            print("NB nominal Lexical Entries with %d" % len(self.lexical_entries.keys()))
            nb_mwe = len([x for x in self.lexical_entries.items() if x[1].is_mwe == True])
            print("   among which %s are MWE" % nb_mwe)
            print("NB Senses of nominal lexical entries %d" % len(self.lexical_senses.keys()))
            print("\n")
            print("AFTER FILTERING")
            print("===================================")
            print(self.pages.keys())
            print(self.lexical_entries.keys())
            print(self.lexical_senses.keys())

            for page_id in self.pages:
                print(page_id, self.pages[page_id].get_entry_ids())
                print(f"{page_id} is multi entries page: {self.pages[page_id].is_multi_entries_page()}")

            print("\n")

            for entry_id in self.lexical_entries:
                print((entry_id, self.lexical_entries[entry_id].get_sense_ids()))
                print(f"{entry_id} is multi sense entry: {self.lexical_entries[entry_id].is_multi_senses_entry()}")
            print("\n")
            self.pretty_print()


class Parser:

    def __init__(self,
                 relations,
                 trace,
                 categories,
                 rdf_types,
                 lang,
                 file="fr_dbnary_ontolex.ttl",
                 wiki=Wiktionary(),
                 wiki_dump="wiki_dump.pkl",
                 read=False,
                 read_and_dump=False,
                 filter=False,
                 dump=None,
                 mode='read',
                 labels_to_ignore=[]):
        self.relations = relations
        self.dump = dump
        self.labels_to_ignore = labels_to_ignore
        self.mode = mode
        self.trace = trace
        self.categories = categories
        self.rdf_types = rdf_types
        self.wiki_dump = wiki_dump
        self.file = file
        self.wiki = wiki
        self.read_ = read
        self.filter_ = filter
        self.read_and_dump_ = read_and_dump
        self.lang = lang

    def read(self):

        with open(self.file, "r", encoding="utf-8") as f:

            # variables initialisation
            e1 = None
            relation = None
            e2 = None
            unallowed_cat = False
            unallowed_type = False
            unallowed_label = False
            last_line_of_paragraph = False
            rdf_type = None

            # dict temp data entry
            entry_temp_data = {}
            entry_temp_data["id"] = None
            entry_temp_data["is_mwe"] = False
            entry_temp_data["lemma"] = None
            entry_temp_data["pos"] = None
            entry_temp_data["synonym"] = set()
            entry_temp_data["cf_id"] = None
            entry_temp_data["sense_id"] = None

            # dict temp data sense
            sense_temp_data = {}
            sense_temp_data["id"] = None
            sense_temp_data["example"] = []
            sense_temp_data["definition"] = None
            sense_temp_data["synonym"] = set()
            sense_temp_data["labels"] = set()

            # read each line of the file
            while 1:
                line = f.readline()
                # premieres lignes definissant des @prefix
                if line.startswith('@'):
                    continue
                # exit from the while if EOF
                if not line:
                    break
                sline = line.strip()

                last_line_of_paragraph = False
                # if the line is not empty
                if sline:

                    # checks if this is the last line of a paragraph
                    if sline.endswith(" ."):  # or sline.endswith("@fr"):
                        last_line_of_paragraph = True

                    # if the line starts with a space or spaces then the e1 is the same as before
                    if line.startswith(" "):

                        # if the line ends with a dot then it is the last line of the paragraph
                        if last_line_of_paragraph:
                            delimiter = ' .'
                        else:
                            delimiter = ' ;'
                        # get the relation and e2
                        relation, e2 = line.strip().split(None, 1)
                        e2 = e2.strip()
                        if e2.startswith('[') and not last_line_of_paragraph:
                            while not e2.endswith('] ;') and not e2.endswith('] .'):
                                e2 += f.readline().strip()
                            if e2.endswith('] .'):
                                last_line_of_paragraph = True
                        e2 = e2.strip(delimiter).split(' , ')
                        e2 = [normalization_id(x, lang=self.lang) for x in e2]

                    # new paragraph
                    elif line.startswith(self.lang + ":") or line.startswith("<http://kaiko.getalp.org/dbnary/fra/"):
                        rdf_types = None
                        unallowed_type = False
                        unallowed_cat = False
                        unallowed_label = False
                        entry_temp_data["id"] = None
                        entry_temp_data["is_mwe"] = False
                        entry_temp_data["lemma"] = None
                        entry_temp_data["pos"] = None
                        entry_temp_data["synonym"] = set()
                        entry_temp_data["cf_id"] = None
                        entry_temp_data["sense_id"] = None
                        sense_temp_data["id"] = None
                        sense_temp_data["example"] = []
                        sense_temp_data["definition"] = None
                        sense_temp_data["synonym"] = set()
                        sense_temp_data["labels"] = set()

                        # if the is no ' ' in the line then there is only the e1 for the paragraph
                        if " " not in line:
                            # get the e1, relation and e2
                            e1 = line.strip()
                            e1 = normalization_id(e1, lang=self.lang)
                            relation = None
                            e2 = []
                        else:
                            # get the e1, relation and e2
                            e1, relation, e2 = line.strip().split(None, 2)
                            e2 = e2.strip()
                            if e2.startswith('[') and not last_line_of_paragraph:
                                while not e2.endswith('] ;') and not e2.endswith('] .'):
                                    e2 += f.readline().strip()
                                if e2.endswith('] .'):
                                    last_line_of_paragraph = True

                            e2 = e2.strip(' ;').strip(' .').split(' , ')
                            e2 = [normalization_id(x, lang=self.lang) for x in e2]
                            e1 = normalization_id(e1, lang=self.lang)
                    else:
                        print("WARNING: %s" % line)

                    # checks if there is a new rdf type for the paragraph
                    if relation == "rdf:type":
                        rdf_types = e2

                    # FILTERING
                    # if this line is an explicit triple
                    if rdf_types is not None:
                        # if the rdf type is note in those of interest then the paragraph is skipped
                        unallowed_type = True
                        for t in rdf_types:
                            if t in self.rdf_types:
                                unallowed_type = False
                                break

                    # checks if this is a line of interest or not
                    # NB: if last_line_of_paragraph, then we should go on even if not allowed_relations, to add the Page,  LE or LS
                    if relation == None or unallowed_type or (
                            relation not in self.relations and not last_line_of_paragraph):
                        continue
                    # print the rdf triplet of the line
                    if self.trace:
                        print("e1: ", e1)
                        print("relation : ", relation)
                        print("e2: ", e2)
                        print("\n")

                    # print("e1: ", e1)
                    # print("relation : ", relation)
                    # print("e2: ", e2[0])
                    # print("\n")
                    # PROCESSING
                    # if rdf type is a page
                    if "dbnary:Page" in rdf_types:
                        page_id = e1
                        if relation == "dbnary:describes":
                            self.wiki.add_page(page_id=page_id, page=Page(id=page_id, entry_ids=e2))

                    # if rdf type is a lexical entry
                    elif "ontolex:LexicalEntry" in rdf_types:
                        entry_temp_data["id"] = e1
                        entry_temp_data["is_mwe"] = True if "ontolex:MultiWordExpression" in rdf_types else False

                        if relation == "rdfs:label":
                            entry_temp_data["lemma"] = e2[0].strip('"').removesuffix('"@fr')
                        if relation == "lexinfo:partOfSpeech":
                            entry_temp_data["pos"] = e2[0]
                            if e2[0] not in self.categories:
                                unallowed_cat = True
                        if relation == "dbnary:partOfSpeech":
                            unallowed_cat = True
                            for cat in self.categories:
                                if e2[0] == cat:
                                    # print("CAT: ", e2[0])
                                    unallowed_cat = False
                        if relation == "dbnary:synonym":
                            for synonym in e2:
                                entry_temp_data["synonym"].add(synonym)
                        if relation == "ontolex:canonicalForm":
                            entry_temp_data["cf_id"] = e2
                        if relation == "ontolex:sense":
                            entry_temp_data["sense_id"] = e2
                        # @@ on n'ajoute que si le filtrage categorie est ok
                        if last_line_of_paragraph and not unallowed_cat:
                            self.wiki.add_lexical_entry(entry_id=entry_temp_data["id"],
                                                        entry=lexicalEntry(id=entry_temp_data["id"],
                                                                           pos=entry_temp_data["pos"],
                                                                           lemma=entry_temp_data["lemma"],
                                                                           sense_ids=entry_temp_data["sense_id"],
                                                                           cf_ids=entry_temp_data["cf_id"],
                                                                           is_mwe=entry_temp_data["is_mwe"],
                                                                           synonyms=entry_temp_data["synonym"])
                                                        )
                            if self.trace:
                                for data in entry_temp_data:
                                    print(data, entry_temp_data[data])

                    # if rdf type is a lexical sense
                    elif "ontolex:LexicalSense" in rdf_types:
                        sense_temp_data["id"] = e1
                        # sense_temp_data["example"] = []
                        if relation == "skos:definition":
                            definition = extract_quote(text=e2[0])
                            # print(definition)
                            labels, definition = extract_labels(text=definition)
                            sense_temp_data["definition"] = definition
                            sense_temp_data["labels"] = labels
                            if labels is not None:
                                for l in labels:
                                    if l in self.labels_to_ignore:
                                        unallowed_label = True
                                        break

                            # TEST EXTRACTION LABELS AND DEFINITION
                            # print("\n")
                            # print(e2[0])
                            # print("DEFINITION: ", extract_quote(text=e2[0]))
                            # print("LABELS: ", labels)
                            # print("\n")
                        if relation == "skos:example":
                            # pass
                            sense_temp_data["example"].append(extract_quote(text=e2[0]))

                            # sense_temp_data["example"].append(e2[0])
                            # TEST EXTRACTION EXAMPLE
                            # print("\n")
                            # print("EXAMPLE: ", extract_quote(text=e2[0], left_delimiter='"', right_delimiter='@'))
                            # print("\n")
                        if relation == "dbnary:synonym":
                            for synonym in e2:
                                sense_temp_data["synonym"].add(synonym)
                        if last_line_of_paragraph and not unallowed_label:
                            sid = sense_temp_data["id"]
                            self.wiki.add_lexical_sense(sense_id=sid,
                                                        sense=lexicalSense(
                                                            id=sid, definition=sense_temp_data["definition"],
                                                            examples=sense_temp_data["example"],
                                                            synonyms=sense_temp_data["synonym"],
                                                            labels=sense_temp_data["labels"])
                                                        )
                            if self.trace:
                                for data in sense_temp_data:
                                    print(data, sense_temp_data[data])

                    # if rdf type is a canonical form
                    elif "ontolex:Form" in rdf_types:
                        if relation == "lexinfo:gender":
                            self.wiki.add_canonical_form(e1, e2[0])
        self.wiki.filter_wiki(trace=self.trace)
        # exit()
        print("READ TERMINATED")
        return self.wiki

    def filter(self):
        filtered_file = open("filtered_fr_dbnary_ontolex.ttl", "w", encoding="utf-8")
        with open(self.file, "r", encoding="utf-8") as f:

            # variables initialisation
            paragraph = []
            skip = False

            # read each line of the file
            while 1:
                line = f.readline()

                # premieres lignes definissant des @prefix
                if line.startswith('@'):
                    continue
                # exit from the while if EOF
                if not line:
                    break

                if line.startswith(self.lang + ":") or line.startswith("<"):
                    if not skip:
                        for l in paragraph:
                            filtered_file.write(l)
                    paragraph = []
                    skip = False
                if "rdf:type" in line:
                    # print("RDF_TYPE")
                    skip = True
                    for rdf_type in self.rdf_types:
                        if rdf_type in line:
                            skip = False
                if 'dbnary:partOfSpeech' in line or 'lexinfo:partOfSpeech' in line:
                    # print("POS")
                    skip = True
                    for cat in self.categories:
                        if cat in line:
                            # print("CATEGORY_"+cat)
                            skip = False

                paragraph.append(line)
        filtered_file.close()

    def read_and_dump(self):
        wiki = self.read()
        # Serialize the object to a file
        with open("wiki_dump.pkl", "wb") as file:
            pickle.dump(wiki, file)

    def parse_file(self):
        if self.read_:
            print("READ")
            self.read()
        elif self.read_and_dump_:
            print("READ AND DUMP")
            self.read_and_dump()
        elif self.filter_:
            print("FILTER")
            self.filter()

    def set_parser_mode(self, mode):
        if mode == "read":
            self.read_ = True
            self.read_and_dump_ = False
            self.filter_ = False

        elif mode == "read_and_dump":
            self.read_ = False
            self.read_and_dump_ = True
            self.filter_ = False

        elif mode == "filter":
            self.read_ = False
            self.read_and_dump_ = False
            self.filter_ = True
