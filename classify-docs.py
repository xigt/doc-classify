#!/usr/bin/env python3
import os, re, pickle
from argparse import ArgumentParser, ArgumentError
from configparser import ConfigParser
from collections import Counter

import sys





# Start by loading the default config file.
from urllib.parse import urlparse

# -------------------------------------------
# Set up the default config file
# -------------------------------------------
my_dir = os.path.dirname(__file__)
defaults_path = os.path.join(my_dir, 'defaults.ini')

class DefaultConfigParser(ConfigParser):
    @classmethod
    def init_from_path(cls, path, defaults=None):
        if defaults is None:
            defaults = {}
        c = cls(defaults=defaults)
        new_c = cls()
        new_c.read(path)
        c.update(new_c.defaults())
        return c

    def get(self, option, *args, **kwargs):
        return super().get('DEFAULT', option, **kwargs)

    def set(self, option, value=None):
        super().set('DEFAULT', option, value)

    def update(self, d):
        for k, v in d.items():
            super().set('DEFAULT', k, v)

defaults = DefaultConfigParser()
if os.path.exists(defaults_path): defaults.read(defaults_path)

# -------------------------------------------
# Process the URL list
# -------------------------------------------
def get_urls(url_path):
    url_mapping = {}
    with open(url_path, 'r') as f:
        for line in f:
            num, url = line.split()
            url_mapping[num] = url
    return url_mapping

# -------------------------------------------
# Process the list of IGT containing docs
# -------------------------------------------
def get_identified_docs(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            yield line.strip()

def get_doc_id(path):
    return re.search('([0-9]+)', os.path.basename(path)).group(1)

# -------------------------------------------
# Get features from the URL
# -------------------------------------------
def get_url_features(url):
    feats = set([])
    p = urlparse(url)
    feats.add('url_domain_' + p.netloc)
    [feats.add('url_domain_elt_' + elt) for elt in p.netloc.split('.')]
    feats.add('url_filename_'+os.path.basename(p.path))

    for elt in os.path.dirname(p.path.strip()).split('/'):
        if elt.strip():
            feats.add('url_path_elt_'+elt)

    return feats



# -------------------------------------------
# Get features for a given document
# -------------------------------------------
def get_doc_features(path, url_dict):
    """
    Given the path to a document and the url dict,
    return a dictionary of {featname:val} pairs.

    :rtype: dict
    """
    doc_id = get_doc_id(path)


    # -- 1) Get the text features
    text_feats = Counter()
    fd = FrekiDoc.read(path)
    for line in fd.lines():
        text_feats.update(['word_'+w.lower() for w in re.findall('\w+', line)])

    feat_dict = dict(text_feats)

    # -- 2) Add URL features
    if url_dict:
        url = url_dict[doc_id]
        url_feats = get_url_features(url)
        feat_dict.update({f:1 for f in url_feats})

    return feat_dict

class ClassifierWrapper(object):
    def __init__(self):
        self.learner = LogisticRegression()
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = SelectKBest(chi2, 10000)
        self.classes = []

    def train(self, labels, data):
        vec = self.dv.fit_transform(data, labels)
        vec = self.feat_selector.fit_transform(vec, labels)

        self.learner.fit(vec, labels)

    def feat_names(self):
        return np.array(self.dv.get_feature_names())

    def selected_feats(self):
        return self.feat_selector.get_support()

    def weights(self):
        return {f:self.learner.coef_[0][j] for j, f in enumerate(self.feat_names()[self.selected_feats()])}

    def print_weights(self, n=1000):
        sorted_weights = sorted(self.weights().items(), reverse=True, key=lambda x: x[1])
        for feat_name, weight in sorted_weights[:n]:
            print('{}\t{}'.format(feat_name, weight))

def recurse_files(dir, dir_regex=None, file_regex=None):
    dir_regex = re.compile(dir_regex) if dir_regex else None
    file_regex = re.compile(file_regex) if file_regex else None

    results = []
    for dirpath, dirnames, filenames in os.walk(dir):
        files = [os.path.join(dirpath, p) for p in filenames if file_regex and re.search(file_regex, p)]
        dirs  = [os.path.join(dirpath, p) for p in dirnames if dir_regex and re.search(dir_regex, p)]

        results.extend(dirs + files)
    return results

# -------------------------------------------
# Find_docs
# -------------------------------------------
def find_docs(doc_root=None, doc_glob=None, label_path=None, **kwargs):

    identified_docs = list(get_identified_docs(label_path))

    labels, measurements = [], []

    cw = ClassifierWrapper()

    paths = recurse_files(doc_root, file_regex=doc_glob)

    for path in paths:
        feats = get_doc_features(path, **kwargs)
        docid = get_doc_id(path)
        if docid in identified_docs:
            labels.append('IGT')
        else:
            labels.append('NONIGT')
        measurements.append(feats)


    vec = cw.dv.fit_transform(measurements, labels)
    vec = cw.feat_selector.fit_transform(vec, labels)

    print(vec.shape)

    #cw.learner.verbose = True
    cw.learner.fit(vec, labels)

    print(cw.learner.coef_.shape)

    feat_names = np.array(cw.dv.get_feature_names())
    selected_feats = cw.feat_selector.get_support()

    weights = {f:cw.learner.coef_[0][j] for j, f in enumerate(feat_names[selected_feats])}
    sorted_weights = sorted(weights.items(), reverse=True, key=lambda x: x[1])

    for feat_name, weight in sorted_weights[:1000]:
        print('{}\t{}'.format(feat_name, weight))

    # Now, save the classifier...
    with open('doc-classifier.model', 'wb') as f:
        pickle.dump(cw, f)

def classify_docs(url_mapping=None, doc_root=None, model_path=None, **kwargs):
    with open(model_path, 'rb') as f:
        cw = pickle.load(f)
        assert isinstance(cw, ClassifierWrapper)

    for doc in recurse_files(doc_root, file_regex='^[0-9]+\.txt$'):
        feats = get_doc_features(doc, url_mapping=url_mapping, **kwargs)

        vec = cw.dv.transform(feats)
        vec = cw.feat_selector.transform(vec)
        if 'IGT' in cw.learner.predict(vec):
            print(doc)

true_opts = {'t','true','1'}
false_opts = {'f', 'false', '0'}
bool_opts = true_opts | false_opts

def get_labels(label_path):
    """
    Take the file that specifies the file labels and return
    a dictionary of {doc_id:label} pairs.
    :rtype: dict
    """
    label_dict = {}

    with open(label_path, 'r') as f:
        for line in f:
            doc_id, label = line.split()
            if label.lower() not in bool_opts:
                raise Exception('Unknown label "{}"'.format(label))
            else:
                label_dict[doc_id] = label.lower() in true_opts
    return label_dict

def get_training_data(root_dir, label_dict, url_dict):
    """
    :rtype:
    """
    labels = []
    data = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            doc_id = get_doc_id(filename)
            if doc_id in label_dict:
                fullpath = os.path.join(dirpath, filename)
                data.append(get_doc_features(fullpath, url_dict))
                labels.append(label_dict[doc_id])

    return labels, data




def train_classifier(config):
    """
    Train the classifier.

    :type config: ConfigParser
    """
    # --1) Get the list of URLs, if it exists
    url_path = config.get('url_list', fallback=None)
    url_dict = get_urls(url_path)

    # --2) Get the labels for the training instances
    label_dict = get_labels(config.get('label_path'))

    labels, data = get_training_data(config.get('doc_root'), label_dict, url_dict)

    cw = ClassifierWrapper()
    cw.train(labels, data)

    cw.print_weights()



# -------------------------------------------
# Parse a config file if it exists.
# -------------------------------------------
def def_cp(path):
    """
    Function to use for the "type=" argument in the argparser to load
    a DefaultConfigParser
    :rtype: DefaultConfigParser
    """
    if not os.path.exists(path):
        raise ArgumentError('The specified path "{}" does not exist.'.format(path))
    else:
        c = DefaultConfigParser.init_from_path(path, defaults=defaults.defaults())
        return c

class ConfigError(Exception): pass

def verify_args(config):
    """
    Sanity-check the arguments and print out any errors.

    :type config: DefaultConfigParser
    """
    def specified(opt, name):
        path = config.get(opt, fallback=None)
        if path is None:
            raise ConfigError("No {} was specified".format(name))

    def exists(opt, name):
        path = config.get(opt, fallback=None)
        if path is not None and not os.path.exists(path):
            raise ConfigError('The specified {} "{}" was not found'.format(name, path))

    def specified_and_exists(opt, name):
        specified(opt, name)
        exists(opt, name)

    specified_and_exists('label_path', 'Label Path')
    exists('url_path', "URL Path")
    specified_and_exists('doc_root', 'document root')

    # Add items to the pythonpath
    for path_elt in config.get('python_paths', fallback='').split(':'):
        if path_elt:
            sys.path.insert(0, path_elt)


if __name__ == '__main__':
    # Set up a main argument parser to add subcommands to.
    main_parser = ArgumentParser()

    # Now, add a parser to handle common things like verbosity.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument('-v', '--verbose', action='store_true')
    common_parser.add_argument('-c', '--config', help='Alternate config', type=def_cp)
    common_parser.add_argument('-f', '--force', help="Overwrite files.")
    common_parser.add_argument('-m', '--model-path', help="Path to classifier model.")

    # Set up the subparsers.
    subparsers = main_parser.add_subparsers(help='Valid subcommands', dest='subcommand')
    subparsers.required = True

    # Training parser
    train_p = subparsers.add_parser('train', parents=[common_parser])

    # Testing parser
    test_p = subparsers.add_parser('test', parents=[common_parser])

    args = main_parser.parse_args()

    # Get the config file,
    config = args.config if args.config is not None else DefaultConfigParser(defaults=defaults.defaults())

    # Overwrite stuff from the config files with commandline args
    # if necessary.
    if args.model_path: config.set('model_path', args.model_path)

    # -------------------------------------------
    # Do error checking of arguments, as well as add
    # any needed paths to PYTHONPATH.
    # -------------------------------------------
    verify_args(config)

    # Import nonstandard libs.
    from freki.serialize import FrekiDoc
    from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
    from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
    from sklearn.linear_model.logistic import LogisticRegression
    import numpy as np
    # -------------------------------------------

    if args.subcommand == 'train':
        train_classifier(config)
    elif args.subcommand == 'test':
        pass
    else:
        raise ArgumentError("Unrecognized subcommand")
