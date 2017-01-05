#!/usr/bin/env python3
import os, re, pickle
import random
import statistics
from argparse import ArgumentParser, ArgumentError
from collections import defaultdict
from configparser import ConfigParser
from collections import Counter
import gzip
import sys
from urllib.parse import urlparse

# -------------------------------------------
# Set up logging
# -------------------------------------------
import logging
LOG = logging.getLogger()
LOG.addHandler(logging.StreamHandler(sys.stdout))


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
        # self.learner = LinearSVC()
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = None
        self.classes = []

    def _vectorize(self, data, testing=False):
        if testing:
            return self.dv.transform(data)
        else:
            return self.dv.fit_transform(data)

    def _vectorize_and_select(self, data, labels, num_feats = 10000, testing=False):
        vec = self._vectorize(data, testing=testing)

        # Only do feature selection if num_feats is positive.
        if num_feats is not None and num_feats > 0:

            # When training, assume the feature selector needs
            # to be initialized.
            if not testing:
                self.feat_selector = SelectKBest(chi2, num_feats)
                return self.feat_selector.fit_transform(vec, labels)
            else:
                return self.feat_selector.transform(vec)
        else:
            return vec

    def train(self, data, num_feats = 10000):
        """
        :type data: list[DocInstance]
        """
        labels = [d.label for d in data]
        feats  = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, num_feats=num_feats, testing=False)
        self.learner.fit(vec, labels)

    def test(self, data):
        """
        :type data: list[DocInstance]
        """
        labels = [d.label for d in data]
        feats  = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, testing=True)
        return self.learner.predict_proba(vec)

    def feat_names(self):
        return np.array(self.dv.get_feature_names())

    def feat_supports(self):
        if self.feat_selector is not None:
            return self.feat_selector.get_support()
        else:
            return np.ones((len(self.dv.get_feature_names())), dtype=bool)

    def weights(self):
        return {f: self.learner.coef_[0][j] for j, f in enumerate(self.feat_names()[self.feat_supports()])}

    def print_weights(self, n=1000):
        sorted_weights = sorted(self.weights().items(), reverse=True, key=lambda x: x[1])
        for feat_name, weight in sorted_weights[:n]:
            print('{}\t{}'.format(feat_name, weight))

    def save(self, path):
        """
        Serialize the classifier out to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = gzip.GzipFile(path, 'w')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):
        f = gzip.GzipFile(path, 'r')
        c = pickle.load(f)
        assert isinstance(c, ClassifierWrapper)
        return c




true_opts = {'t','true','1'}
false_opts = {'f', 'false', '0'}
bool_opts = true_opts | false_opts

def true_val(s):
    return s.lower() in true_opts

def get_labels(label_path):
    """
    Take the file that specifies the file labels and return
    a dictionary of {doc_id:label} pairs.
    :rtype: dict
    """
    label_dict = {}

    with open(label_path, 'r') as f:
        for line in f:
            doc_id, label = line.split()[:2]
            if label.lower() not in bool_opts:
                raise Exception('Unknown label "{}"'.format(label))
            else:
                label_dict[doc_id] = label.lower() in true_opts

    counts = Counter(label_dict.values())

    LOG.debug("Loaded label dict with {} positive examples, {} negative.".format(
        counts.get(True), counts.get(False)
    ))

    return label_dict

class DocInstance(object):
    """
    Document wrapper
    """
    def __init__(self, doc_id, label, feats, path):
        self.path = path
        self.doc_id = doc_id
        self.label = label
        self.feats = feats

    def get_feats(self, url_dict = None):
        self.feats = get_doc_features(self.path, url_dict)


def get_training_data(root_dir, label_dict, url_dict):
    """
    :rtype: list[DocInstance]
    """
    data = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            doc_id = get_doc_id(filename)
            if doc_id in label_dict:
                fullpath = os.path.join(dirpath, filename)
                feats = get_doc_features(fullpath, url_dict)
                label = label_dict[doc_id]
                d = DocInstance(doc_id, label, feats, fullpath)
                d.get_feats(url_dict=url_dict)
                data.append(d)

    return data




def train_classifier(argdict):
    """
    Train the classifier.

    :type argdict: dict
    """
    # --1) Get the list of URLs, if it exists
    LOG.info("Obtaining URL list...")
    url_path = argdict.get('url_list', None)
    url_dict = get_urls(url_path)

    # --2) Get the labels for the training instances
    LOG.info("Obtaining label list...")
    label_dict = get_labels(argdict.get('label_path'))

    # --3) Extract features from documents with known labels
    LOG.debug("Extracting training data...")
    data = get_training_data(argdict.get('doc_root'), label_dict, url_dict)

    LOG.info("Beginning training...")

    # --5) Split the training data according to the
    #      nfold training ratio.

    iterations = int(argdict.get('nfold_iterations', '1'))

    # Split data starting at this index...
    train_ratio = float(argdict.get('train_ratio', 1))
    split_index = int(len(data) * train_ratio)

    # Shift data by this much each iteration.
    split_window = int(len(data) * (1 / iterations))

    seed = argdict.get('random_seed')

    r = random.Random()
    if seed is not None: r.seed(int(seed))

    # Shuffle the data
    r.shuffle(data)



    LOG.info('Running {} iterations with {} training portion.'.format(
        iterations, train_ratio))

    overall_stats = StatDict()
    iteration_stats = StatDictList()

    for iter in range(iterations):
        train_portion = data[:split_index]
        test_portion = data[split_index:]

        # Rename the model by iteration if needed
        model_path = argdict.get('model_path', 'model.model')
        if iterations > 1:
            model_base, model_ext = os.path.splitext(model_path)
            iter_model_path = '{}_{}{}'.format(model_base, iter, model_ext)
        else:
            iter_model_path = model_path

        LOG.info('Training model {} with {} instances'.format(iter_model_path, len(train_portion)))
        LOG.debug('Training IDs: {}'.format(','.join(['{}:{}'.format(d.doc_id, d.label) for d in train_portion])))

        # -------------------------------------------
        # Create the classifier wrapper, train and save.
        # -------------------------------------------
        cw = ClassifierWrapper()
        cw.train(train_portion, num_feats=40000)
        cw.save(iter_model_path)
        # -------------------------------------------

        if train_ratio < 1.0:

            LOG.info("Testing model {} on {} instances".format(iter_model_path, len(test_portion)))
            LOG.debug('Testing IDs: {}'.format(','.join([d.doc_id for d in test_portion])))

            gold_labels = [d.label for d in test_portion]
            test_probs = cw.test(test_portion)

            test_labels = []
            # Get the test labels, filtering by threshold
            acceptance_thresh = float(argdict.get('acceptance_thresh', 0.5))

            # Get the index for the "true" class
            t_idx = cw.learner.classes_.tolist().index(True)

            for distribution in test_probs:
                if distribution[t_idx] > acceptance_thresh:
                    test_labels.append(True)
                else:
                    test_labels.append(False)

            # Calculate stats
            iter_dict = StatDict()
            iter_dict.add_iter(test_labels, gold_labels)
            overall_stats.update(iter_dict)

            LOG.info("Iteration accuracy: {:.2f}".format(iter_dict.accuracy()))
            iteration_stats.append(iter_dict)

            LOG.info("Iteration P/R/F for positives: {}".format(iter_dict.prf_string(True)))

            # Shift the data by the training window...
            data = data[split_window:] + data[:split_window]

    if train_ratio < 1.0:
        LOG.info('Overall positive P/R/F: {}'.format(overall_stats.prf_string(True)))
        LOG.info('Overall positive P/R/F Std.dev: {}'.format(iteration_stats.prf_stdev(True)))
        LOG.info('Overall accuracy: {:.2f}'.format(overall_stats.accuracy()))
        LOG.info('Accuracy Std. dev: {:.2f}'.format(iteration_stats.a_stdev()))
    else:
        LOG.info('Training portion was 100%. No testing performed.')

class StatDict():
    def __init__(self):
        self._predictdict = defaultdict(int)
        self._golddict = defaultdict(int)
        self._matchdict = defaultdict(int)
        self._classes = set([])

    def add(self, predicted, gold, n=1):
        self._predictdict[predicted] += n
        self._golddict[gold] += n
        if predicted == gold:
            self._matchdict[gold] += n

        self._classes.add(predicted)
        self._classes.add(gold)

    def add_iter(self, predicted, gold):
        assert len(predicted) == len(gold)
        for p, g in zip(predicted, gold):
            self.add(p, g)

    def precision(self, cls):
        num = self._matchdict[cls]
        denom = self._predictdict[cls]
        return num/denom if denom != 0 else 0

    def recall(self, cls):
        num   = self._matchdict[cls]
        denom = self._golddict[cls]
        return num / denom if denom != 0 else 0

    def accuracy(self):
        matches = sum(self._matchdict.values())
        golds = sum(self._golddict.values())
        return matches/golds * 100

    def fmeasure(self, cls):
        num = 2*(self.precision(cls)*self.recall(cls))
        denom = (self.precision(cls)+self.recall(cls))
        return num/denom if denom != 0 else 0

    def update(self, o):
        """
        :type o: StatDict
        """
        assert isinstance(o, StatDict)
        for k in o._classes:
            self._golddict[k] += o._golddict[k]
            self._predictdict[k] += o._predictdict[k]
            self._matchdict[k] += o._matchdict[k]
            self._classes.add(k)

    def prf_string(self, cls):
        return "{:.2f}/{:.2f}/{:.2f}".format(self.precision(cls), self.recall(cls), self.fmeasure(cls))

class StatDictList(list):
    def accuracies(self): return [l.accuracy() for l in self]
    def precisions(self, cls): return [l.precision(cls) for l in self]
    def recalls(self, cls): return [l.recall(cls) for l in self]
    def fmeasures(self, cls): return [l.fmeasure(cls) for l in self]
    def p_stdev(self, cls): return statistics.stdev(self.precisions(cls))
    def r_stdev(self, cls): return statistics.stdev(self.recalls(cls))
    def f_stdev(self, cls): return statistics.stdev(self.fmeasures(cls))
    def a_stdev(self): return statistics.stdev(self.accuracies())
    def prf_stdev(self, cls): return '{:.2f}/{:.2f}/{:.2f}'.format(self.p_stdev(cls), self.r_stdev(cls), self.f_stdev(cls))

def analyze_stats(test_labels, gold_labels):
    """
    Function to give analysis of model
    """
    assert len(test_labels) == len(gold_labels), "Length of label lists disagree"

    match_dict = {True:0, False:0}
    test_dict = {True:0, False:0}
    gold_counts = {True:0, False:0}

    total_matches = 0

    for test_label, gold_label in zip(test_labels, gold_labels):
        if test_label == gold_label:
            match_dict[test_label] += 1
            total_matches += 1
        gold_counts[gold_label] += 1
        test_dict[test_label] += 1

    accuracy = total_matches / len(test_labels) * 100
    pos_precision = match_dict[True] / test_dict[True]
    pos_recall = match_dict[True] / gold_counts[True]
    pos_fmeasure = 2*(pos_precision+pos_recall)/(pos_precision * pos_recall)

    return accuracy, pos_precision, pos_recall, pos_fmeasure

def test_classifier(argdict):
    """
    :type argdict: dict
    """

    # Load the classifier...
    LOG.info("Loading model...")
    model_path = argdict.get('model_path')
    ClassifierWrapper.load(model_path)



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

def process_args(argdict):
    """
    Sanity-check the arguments and print out any errors.

    :type argdict: dict
    """
    def specified(opt, name):
        path = argdict.get(opt, None)
        if path is None:
            raise ConfigError("No {} was specified".format(name))

    def exists(opt, name):
        path = argdict.get(opt, None)
        if path is not None and not os.path.exists(path):
            raise ConfigError('The specified {} "{}" was not found'.format(name, path))

    def specified_and_exists(opt, name):
        specified(opt, name)
        exists(opt, name)

    specified_and_exists('label_path', 'Label Path')
    exists('url_path', "URL Path")
    specified_and_exists('doc_root', 'document root')

    # Add items to the pythonpath
    for path_elt in argdict.get('python_paths', '').split(':'):
        if path_elt:
            sys.path.insert(0, path_elt)

    # Handle verbosity
    verbosity = int(argdict.get('verbose', 0))
    if verbosity == 1:
        LOG.setLevel(logging.INFO)
    elif verbosity == 2:
        LOG.setLevel(logging.DEBUG)



if __name__ == '__main__':
    # Set up a main argument parser to add subcommands to.
    main_parser = ArgumentParser()

    # Now, add a parser to handle common things like verbosity.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument('-v', '--verbose', action='count')
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

    # Merge config file and commandline arguments into one dict.
    argdict = config.defaults()
    for k, v in vars(args).items():
        if v is not None:
            argdict[k] = v

    # -------------------------------------------
    # Do error checking of arguments, as well as add
    # any needed paths to PYTHONPATH.
    # -------------------------------------------
    process_args(argdict)

    # Import nonstandard libs.
    from freki.serialize import FrekiDoc
    from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
    from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.svm import LinearSVC, LinearSVR, OneClassSVM
    import numpy as np
    # -------------------------------------------

    if args.subcommand == 'train':
        train_classifier(argdict)
    elif args.subcommand == 'test':
        test_classifier(argdict)
    else:
        raise ArgumentError("Unrecognized subcommand")
