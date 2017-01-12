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
from gzip import GzipFile
from urllib.parse import urlparse

# -------------------------------------------
# Set up logging
# -------------------------------------------
import logging

from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.feature_selection.mutual_info_ import mutual_info_classif

LOG = logging.getLogger()
NORM_LEVEL = 40
logging.addLevelName(NORM_LEVEL, 'NORM')
stdout_handler = logging.StreamHandler(sys.stdout)


stdout_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
LOG.addHandler(stdout_handler)

def normlog(msg):
    LOG.log(NORM_LEVEL, msg)

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
        return super().get(DEFAULT_STR, option, **kwargs)

    def set(self, option, value=None):
        super().set(DEFAULT_STR, option, value)

    def update(self, d):
        for k, v in d.items():
            super().set(DEFAULT_STR, k, v)


defaults = DefaultConfigParser()
if os.path.exists(defaults_path): defaults.read(defaults_path)


# -------------------------------------------
# Process the URL list
# -------------------------------------------
def get_urls(url_path):
    url_mapping = {}

    # Either open the url list as txt or gzip, depending
    # on the extension.
    if url_path.endswith('.gz'):
        f = GzipFile(url_path, 'rb')
    else:
        f = open(url_path, 'rb')

    for line in f:
        line = line.decode()
        num, url = line.split()
        url_mapping[num] = url

    f.close()
    return url_mapping


# -------------------------------------------
# Process the list of IGT containing docs
# -------------------------------------------
def get_identified_docs(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            yield line.strip()


def get_doc_id(path):
    return re.search('([0-9]+)(?:\.freki(?:\.gz)?)', os.path.basename(path)).group(1)


# -------------------------------------------
# Get features from the URL
# -------------------------------------------
def get_url_features(url):
    feats = set([])
    p = urlparse(url)
    feats.add('url_domain_' + p.netloc)
    [feats.add('url_domain_elt_' + elt) for elt in p.netloc.split('.')]
    [feats.add('url_filename_elt_' + elt) for elt in re.split('\_\-\.', os.path.basename(p.path))]

    for elt in os.path.dirname(p.path.strip()).split('/'):
        if elt.strip():
            feats.add('url_path_elt_' + elt)

    return feats


# -------------------------------------------
# Get features for a given document
# -------------------------------------------
def get_doc_features(path, url_dict, lang_set):
    """
    Given the path to a document and the url dict,
    return a dictionary of {featname:val} pairs.

    :rtype: dict
    """
    doc_id = get_doc_id(path)

    # Keep track of the set of pages that have been seen
    # so we can know how many there were in the original doc.
    pages = set([])
    fonts = set([])

    word_lengths = []

    # -- 1) Get the text features
    text_feats = Counter()
    fd = FrekiDoc.read(path)

    use_bigrams = true_val(argdict.get(USE_BIGRAMS, False))

    num_unknowns = 0

    for l_i, line in enumerate(fd.lines()):

        c = Counter(line)
        num_unknowns += c.get('\uFFFD', 0)

        # Operate on each of the words of the sentence.
        words = re.findall('\w+', line, flags=re.UNICODE)
        words = [w.lower() for w in words]
        for w_i, word in enumerate(words):
            word_lengths.append(len(word))

            # Add words as special feature if in the range
            # where it might be part of the title or author.
            if l_i < 10:
                text_feats['title_word_{}'.format(word)] = 1

            # Check whether a word is a language name in
            # the text or title
            if lang_set and word in lang_set:
                text_feats['has_langname'] = 1
                if l_i < 10:
                    text_feats['has_lang_name_in_title'] = 1
                if w_i > 0:
                    if words[w_i - 1] == 'in':
                        text_feats['has_in_langname'] = 1
                        if l_i < 10:
                            text_feats['has_in_langname_in_title'] = 1

            # Add bigrams as features if enabled.
            if use_bigrams and w_i < len(words) - 2:
                w_j = words[w_i + 1]
                text_feats['bigram_{}_{}'.format(w_i, w_j)] = 1
                if l_i < 10:
                    text_feats['title_bigram_{}_{}'.format(w_i, w_j)] = 1

        text_feats.update(['word_' + w for w in words])

        # Get the pages and fonts once so it
        # doesn't have to be done a second time.
        pages.add(line.block.page)
        [fonts.add(f) for f in line]

    feat_dict = dict(text_feats)

    if num_unknowns > 10:
        feat_dict['many_unknown_chars'] = 1

    # -- 2) Add URL features
    if url_dict:
        url = url_dict[doc_id]
        url_feats = get_url_features(url)
        feat_dict.update({f:1 for f in url_feats})

    # -- 3) Add other features
    if len(word_lengths) > 1:
        avg_wd_length = statistics.mean(word_lengths)
        feat_dict['long_words'] = avg_wd_length > 10
        feat_dict['very_long_words'] = avg_wd_length > 20

    # Are there a ton of fonts in the doc?
    num_fonts = len(fonts)
    feat_dict['one_font'] = num_fonts == 1
    feat_dict['few_fonts'] = 1 > num_fonts > 10
    feat_dict['many_fonts'] = num_fonts > 10

    # Is the length of the document
    feat_dict['gt_4_pages'] = len(pages) > 4
    feat_dict['gt_8_pages'] = len(pages) > 8

    return feat_dict


class ClassifierWrapper(object):
    def __init__(self):
        # self.learner = LogisticRegression()
        self.learner = AdaBoostClassifier()
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = None
        self.classes = []

    def _vectorize(self, data, testing=False):
        if testing:
            return self.dv.transform(data)
        else:
            return self.dv.fit_transform(data)

    def _vectorize_and_select(self, data, labels, num_feats=None, testing=False):

        # Start by vectorizing the data.
        vec = self._vectorize(data, testing=testing)

        # Next, filter the data if in testing mode, according
        # to whatever feature selector was defined during
        # training.
        if testing:
            if self.feat_selector is not None:
                # LOG.info('Feature selection was enabled during training, limiting to {} features.'.format(
                #     self.feat_selector.k))
                return self.feat_selector.transform(vec)
            else:
                return vec

        # Only do feature selection if num_feats is positive.
        elif num_feats is not None and (num_feats > 0):
            LOG.info('Feature selection enabled, limiting to {} features.'.format(num_feats))
            self.feat_selector = SelectKBest(chi2, num_feats)
            return self.feat_selector.fit_transform(vec, labels)

        else:
            LOG.info("Feature selection disabled, all available features are used.")
            return vec

    def train(self, data, num_feats=None, weight_path=None):
        """
        :type data: list[DocInstance]
        """
        labels = [d.label for d in data]
        feats = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, num_feats=num_feats, testing=False)
        self.learner.fit(vec, labels)
        if weight_path is not None:
            LOG.info('Writing feature weights to "{}"'.format(weight_path))
            self.dump_weights(weight_path)

    def test(self, data):
        """
        Given a list of document instances, return a list
        of the probabilities of the Positive, Negative examples.

        :type data: list[DocInstance]
        :rtype: list[tuple[float, float]]
        """
        labels = [d.label for d in data]
        feats = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, testing=True)

        # Get the index of the True value...
        t_idx = self.learner.classes_.tolist().index(True)
        f_idx = 0 if t_idx == 1 else 1

        # Now, make sure the returned list is always [P(True), P(False)]
        probs = [(p[t_idx], p[f_idx]) for p in self.learner.predict_proba(vec)]

        return probs

    def feat_names(self):
        return np.array(self.dv.get_feature_names())

    def feat_supports(self):
        if self.feat_selector is not None:
            return self.feat_selector.get_support()
        else:
            return np.ones((len(self.dv.get_feature_names())), dtype=bool)

    def weights(self):
        return {f: self.learner.coef_[0][j] for j, f in enumerate(self.feat_names()[self.feat_supports()])}

    def dump_weights(self, path, n=-1):
        with open(path, 'w') as f:
            sorted_weights = sorted(self.weights().items(), reverse=True, key=lambda x: x[1])
            for feat_name, weight in sorted_weights[:n]:
                f.write('{}\t{}\n'.format(feat_name, weight))

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


true_opts = {'t', 'true', '1', 'on'}
false_opts = {'f', 'false', '0', 'off'}
bool_opts = true_opts | false_opts


def true_val(s):
    return str(s).lower() in true_opts


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

    LOG.info("Loaded label dict with {} positive examples, {} negative.".format(
        counts.get(True), counts.get(False)
    ))

    return label_dict


# =============================================================================
# Training/Testing abstractions
#
# This section deals with creating a class and methods to simplify selecting
# documents for testing and training and passing the documents to the classifier.
# =============================================================================

class DocInstance(object):
    """
    Wrapper class to hold the path, doc_id, label, and features
    """

    def __init__(self, doc_id, label, feats, path):
        self.path = path
        self.doc_id = doc_id
        self.label = label
        self.feats = feats


def get_freki_files(data_dirs):
    """
    Walk through the given list of directories and return
    a list of the freki files.

    :param data_dirs: A colon-separated string of directories
    :rtype: list[str]
    """
    file_list = []
    for root_dir in data_dirs.split(':'):
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.freki') or filename.endswith('.freki.gz'):
                    file_list.append(os.path.join(dirpath, filename))
    return file_list

def get_data(data_dirs, url_dict, lang_set, label_dict=None, training=False):
    """
    Get a list of doc instances (with extracted features) for a
    given list of data directories.

    :rtype: GeneratorType[DocInstance]
    """
    freki_path_list = get_freki_files(data_dirs)
    for freki_path in freki_path_list:
        doc_id = get_doc_id(freki_path)
        if not training or doc_id in label_dict:
            feats = get_doc_features(freki_path, url_dict, lang_set)
            label = label_dict[doc_id] if label_dict else None
            d = DocInstance(doc_id, label, feats, freki_path)
            yield d


def get_training_data(root_dir, url_dict, lang_set, label_dict):
    return get_data(root_dir, url_dict, lang_set, label_dict=label_dict, training=True)

def get_testing_data(root_dir, url_dict, lang_set):
    return get_data(root_dir, url_dict, lang_set, training=False)


# =============================================================================
# Training/Testing Procedures.
# =============================================================================
format_widths = (8, 10.3, 10.3)
format_string = '{{:{}s}}{{:{}g}}{{:{}g}}'.format(*format_widths)
header_string = '{{:>{}s}}{{:>{}s}}{{:>{}s}}'.format(*[int(i) for i in format_widths]).format(
    'doc_id', 'prob_t', 'prob_f')

def format_output(doc_id, prob_t, prob_f):
    return format_string.format(doc_id, prob_t, prob_f)

def write_output(stream, doc_id, prob_t, prob_f):
    stream.write(format_output(doc_id, prob_t, prob_f)+'\n')


def test_classifier(argdict):
    """
    :type argdict: dict
    """
    # --0) Attempt to open the file to write classifications.
    test_output_dir = argdict.get(TEST_OUTPUT)
    try:
        os.makedirs(test_output_dir, exist_ok=True)
    except OSError:
        LOG.critical('Directory "{}" could not be created.'.format(test_output_dir))
        sys.exit(2)

    # --1) Load the classifier...
    normlog("Loading model...")
    model_path = argdict.get(MODEL_PATH)
    cw = ClassifierWrapper.load(model_path)

    # --2) Load the URL dict...
    url_dict = load_urldict(argdict.get(URL_PATH))

    # --2b) Load the lang list...
    lang_set = load_langset(argdict.get(LANG_PATH))

    # --3) Get the files to be classified...
    data_iter = get_testing_data(argdict.get(TEST_DIRS), url_dict, lang_set)


    # --4) Classify the testing data
    #      one instance at a time, and
    #      write out results iteratively.
    results_file = open(os.path.join(test_output_dir,
                                     'all_classifications.txt'),'w')
    pos_docs_file = open(os.path.join(test_output_dir,
                                 'positive_docs.txt'), 'w')

    results_file.write(header_string+'\n')
    pos_docs_file.write(header_string+'\n')

    # -------------------------------------------
    LOG.info("Writing out classifications...")
    for datum in data_iter:
        test_distributions = cw.test([datum])

        acceptance_thresh = argdict.get(ACCEPTANCE_THRESH)
        prob_t, prob_f = test_distributions[0]

        # Now, write out the classification as it happens as
        #  doc_id  Prob(t)    Prob(f)
        write_output(results_file, datum.doc_id, prob_t, prob_f)
        results_file.flush()

        if prob_t > acceptance_thresh:
            write_output(pos_docs_file, datum.doc_id, prob_t, prob_f)

    results_file.close()
    pos_docs_file.close()

    LOG.info("Testing Complete.")


def train_classifier(argdict):
    """
    Train the classifier.

    :type argdict: dict
    """
    # --1a) Get the list of URLs
    url_dict = load_urldict(argdict.get(URL_PATH))

    # --1b) Get the list of langs
    lang_set = load_langset(argdict.get(LANG_PATH))

    # --2) Get the labels for the training instances
    normlog("Obtaining label list...")
    label_dict = get_labels(argdict.get('label_path'))

    # --3) Extract features from documents with known labels
    LOG.debug("Extracting training data...")
    data = list(get_training_data(argdict.get(TRAIN_DIRS), url_dict, lang_set, label_dict))

    LOG.info("Beginning training...")

    # --5) Split the training data according to the
    #      nfold training ratio.

    iterations = argdict.get(NFOLD_ITERS)

    # Split data starting at this index...
    train_ratio = argdict.get(TRAIN_RATIO)
    split_index = int(len(data) * train_ratio)

    # Shift data by this much each iteration.
    split_window = int(len(data) * (1 / iterations))

    seed = argdict.get(RAND_SEED)

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
        model_path = argdict.get(MODEL_PATH, 'model.model')
        if iterations > 1:
            model_base, model_ext = os.path.splitext(model_path)
            iter_model_path = '{}_{}{}'.format(model_base, iter, model_ext)
        else:
            iter_model_path = model_path

        LOG.info('Training model {} with {} instances'.format(iter_model_path, len(train_portion)))
        LOG.debug('Positive Docs: {}'.format(','.join(['{}:{}'.format(d.doc_id, d.label) for d in train_portion if d.label is True])))

        # -------------------------------------------
        # Create the classifier wrapper, train and save.
        # -------------------------------------------
        cw = ClassifierWrapper()

        if true_val(argdict.get(DEBUG)):
            weight_path = os.path.splitext(iter_model_path)[0]+'_weights.txt'
        else:
            weight_path = None

        cw.train(train_portion,
                 num_feats=argdict.get(NUM_FEATS),
                 weight_path=weight_path)

        # Uncomment this out to save all N iterations
        # of the classifier.
        # cw.save(iter_model_path)
        # -------------------------------------------

        if train_ratio < 1.0:

            LOG.info("Testing model {} on {} instances".format(iter_model_path, len(test_portion)))
            LOG.debug('Testing IDs: {}'.format(','.join([d.doc_id for d in test_portion])))

            gold_labels = [d.label for d in test_portion]
            test_probs = cw.test(test_portion)

            test_labels = []
            # Get the test labels, filtering by threshold
            acceptance_thresh = argdict.get(ACCEPTANCE_THRESH, 0.5)

            # Get the index for the "true" class
            t_idx = cw.learner.classes_.tolist().index(True)

            for prob_t, prob_f in test_probs:
                test_labels.append(prob_t > acceptance_thresh)

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


def load_urldict(url_path):
    if url_path is not None and os.path.exists(url_path):
        normlog("Parsing URL list...")
        return get_urls(url_path)
    return {}


def load_langset(lang_path):
    """
    :param lang_path: Path to list of languages
    :rtype: set
    """
    normlog('Loading language list...')
    lang_set = set([])
    if lang_path and os.path.exists(lang_path):
        with open(lang_path, 'r', encoding='utf-8') as f:
            for line in f:
                lang_set.add(line.strip().lower())
    return lang_set


# =============================================================================
# Gathering Statistics
# =============================================================================

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
        return num / denom if denom != 0 else 0

    def recall(self, cls):
        num = self._matchdict[cls]
        denom = self._golddict[cls]
        return num / denom if denom != 0 else 0

    def accuracy(self):
        matches = sum(self._matchdict.values())
        golds = sum(self._golddict.values())
        return matches / golds * 100

    def fmeasure(self, cls):
        num = 2 * (self.precision(cls) * self.recall(cls))
        denom = (self.precision(cls) + self.recall(cls))
        return num / denom if denom != 0 else 0

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

    def prf_stdev(self, cls): return '{:.2f}/{:.2f}/{:.2f}'.format(self.p_stdev(cls), self.r_stdev(cls),
                                                                   self.f_stdev(cls))


def analyze_stats(test_labels, gold_labels):
    """
    Function to give analysis of model
    """
    assert len(test_labels) == len(gold_labels), "Length of label lists disagree"

    match_dict = {True: 0, False: 0}
    test_dict = {True: 0, False: 0}
    gold_counts = {True: 0, False: 0}

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
    pos_fmeasure = 2 * (pos_precision + pos_recall) / (pos_precision * pos_recall)

    return accuracy, pos_precision, pos_recall, pos_fmeasure


# =============================================================================
# Configuration Files
# =============================================================================


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

    errors = []
    warnings = []

    def specified(opt, warn=False, msg=None):
        nonlocal errors, warnings
        use_list = warnings if warn else errors
        path = argdict.get(opt, '')
        if not path.strip():
            use_msg = 'Option "{}" was not specified' if msg is None else msg
            use_list.append(use_msg.format(opt))
            return False
        return True

    def _exist_path(path, opt, warn=False, msg=None):
        nonlocal errors, warnings
        use_list = warnings if warn else errors
        if not os.path.exists(path):
            use_msg = 'The path "{}" for option "{}" was not found' if msg is None else msg
            use_list.append(use_msg.format(path, opt))
            return False
        return True

    def exists(opt, warn=False, msg=None):
        path = argdict.get(opt, '')
        return _exist_path(path, opt, warn, msg)

    def exists_list(opt, warn=False, msg=None):
        error_found = False
        for path in argdict.get(opt, '').split(':'):
            error_found |= _exist_path(path, opt, warn, msg)
        return error_found

    def specified_and_exists(opt, warn=False, spec_msg=None, exist_msg=None):
        specified(opt, warn, spec_msg) and exists(opt, warn, exist_msg)

    def specified_and_exists_list(opt, warn=False, spec_msg=None, exist_msg=None):
        specified(opt, warn, spec_msg) and exists_list(opt, warn, exist_msg)

    # Extract the subcommand to do conditional checks.
    subcommand = argdict.get('subcommand')

    if subcommand == TRAIN_CMD:
        specified_and_exists('label_path')
        specified_and_exists_list(TRAIN_DIRS)
    elif subcommand == TEST_CMD:
        specified_and_exists_list(TEST_DIRS)
        specified(TEST_OUTPUT)
    elif subcommand == NFOLD_ITERS:
        specified(NFOLD_ITERS)
        specified(TRAIN_RATIO)

    specified_and_exists(LANG_PATH, warn=True,
                         spec_msg='Option "{}" was not specified. No language features will be used.',
                         exist_msg='Path "{}" for option "{}" was not specified. No language features will be used.')
    specified_and_exists(URL_PATH, warn=True,
                         spec_msg='Option "{}" was not specified. No URL features will be used.',
                         exist_msg='Path "{}" for option "{}" was not specified. No URL features will be used.')
    specified_and_exists(MODEL_PATH)

    for warning in warnings:
        LOG.warning('{}'.format(warning))

    if errors:
        LOG.critical('Errors encountered running command: "{}"'.format(' '.join(sys.argv)))
        for error in errors:
            LOG.critical(error)
        sys.exit(2)



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

    # Parse ints if needed
    def parse(arg, func=int, default=None):
        if argdict.get(arg) is not None:
            argdict[arg] = func(argdict.get(arg))
        elif default is not None:
            argdict[arg] = default

    parse(NUM_FEATS, int)
    parse(RAND_SEED, int)
    parse(NFOLD_ITERS, int, 1)
    parse(TRAIN_RATIO, float, 1.0)
    parse(ACCEPTANCE_THRESH, float, 0.5)

# -------------------------------------------
# String constants to use in place of literals
# to prevent typos.
# -------------------------------------------
ACCEPTANCE_THRESH = 'acceptance_thresh'
DEBUG = 'debug'
LANG_PATH = 'lang_list'
MODEL_PATH = 'model_path'
NFOLD_ITERS = 'nfold_iterations'
NUM_FEATS   = 'num_features'
RAND_SEED   = 'random_seed'
TEST_DIRS = 'test_dirs'
TEST_OUTPUT = 'test_output_dir'
TRAIN_DIRS = 'train_dirs'
TRAIN_RATIO = 'train_ratio'
URL_PATH = 'url_list'
USE_BIGRAMS = 'use_bigrams'

DEFAULT_STR = 'DEFAULT'

NFOLD_CMD = 'nfold'
TEST_CMD  = 'test'
TRAIN_CMD = 'train'
TRAINTEST_CMD = 'traintest'
# -------------------------------------------


if __name__ == '__main__':
    # Set up a main argument parser to add subcommands to.
    main_parser = ArgumentParser()

    # Now, add a parser to handle common things like verbosity.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument('-v', '--verbose', action='count', help='Increase verbosity of output.')
    common_parser.add_argument('-c', '--config', help='Alternate config', type=def_cp)
    common_parser.add_argument('-d', '--debug', help='Turn on debugging output', action='store_true', default=None)
    common_parser.add_argument('-f', '--force', help="Overwrite files.", action='store_true', default=None)
    common_parser.add_argument('-m', '--model-path', help="Path to classifier model.")

    # Set up the subparsers.
    subparsers = main_parser.add_subparsers(help='Valid subcommands', dest='subcommand')
    subparsers.required = True

    # -------------------------------------------
    # Subcommand parsers
    # -------------------------------------------

    # Train_parser, for train, traintest, and nfold
    train_parent_parser = ArgumentParser(add_help=False)
    train_parent_parser.add_argument('--train-dirs', help='Colon-separated list of directories from which to draw training docs.')
    train_parent_parser.add_argument('--num-features', help='Number of features to limit training the model with.')
    train_parent_parser.add_argument('--label-path', help='Path to the label file to use for model training.')

    # Test_parser, for train, traintest
    test_parent_parser = ArgumentParser(add_help=False)
    test_parent_parser.add_argument('--test-output-dir', help='Output directory to place classification output.')
    test_parent_parser.add_argument('--test-dirs', help='Colon-separated list of directories from which to pull test files.')

    # Training
    train_parser = subparsers.add_parser(TRAIN_CMD, parents=[common_parser, train_parent_parser])

    # Testing parser
    test_p = subparsers.add_parser(TEST_CMD, parents=[common_parser, test_parent_parser])

    # train/test
    traintest_p = subparsers.add_parser(TRAINTEST_CMD, parents=[common_parser,
                                                                train_parent_parser,
                                                                test_parent_parser])

    # Nfold cross-validation parser
    nfold_p = subparsers.add_parser(NFOLD_CMD, parents=[common_parser, train_parent_parser])
    nfold_p.add_argument('--train-ratio', help='Ratio of training data to testing data, as a decimal between (0-1].', type=float)
    nfold_p.add_argument('--nfold-iterations', help='Number of train/test iterations to perform.', type=int)
    nfold_p.add_argument('--random-seed', help='Integer to fix randomization of data shuffling between program invocations.')

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

    # Import nonstandard libs
    from freki.serialize import FrekiDoc
    from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
    from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
    from sklearn.linear_model.logistic import LogisticRegression
    import numpy as np

    # -------------------------------------------

    if args.subcommand == TRAIN_CMD:
        # Perform the same task as nfold,
        # but without multiple iterations.
        argdict[TRAIN_RATIO] = 1.0
        argdict[NFOLD_ITERS] = 1
        train_classifier(argdict)
    elif args.subcommand == TEST_CMD:
        test_classifier(argdict)
    elif args.subcommand == TRAINTEST_CMD:
        argdict[TRAIN_RATIO] = 1.0
        argdict[NFOLD_ITERS] = 1
        train_classifier(argdict)
        test_classifier(argdict)
    elif args.subcommand == NFOLD_CMD:
        # Perform n-fold cross-validation on
        # the training data.
        train_classifier(argdict)
    else:
        raise ArgumentError("Unrecognized subcommand")

