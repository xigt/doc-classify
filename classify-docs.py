#!/usr/bin/env python3
import os, re, pickle
from argparse import ArgumentParser
from configparser import ConfigParser
from collections import Counter

from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np

from freki.serialize import FrekiDoc


# Start by loading the default config file.
from urllib.parse import urlparse

# -------------------------------------------
# Set up the default config file
# -------------------------------------------
my_dir = os.path.dirname(__file__)
defaults_path = os.path.join(my_dir, 'defaults.ini')

defaults = ConfigParser()
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
def get_doc_features(path, **kwargs):
    url_mappings = kwargs.get('url_mappings')
    doc_id = get_doc_id(path)
    url = url_mappings[doc_id]
    url_feats = get_url_features(url)

    text_feats = Counter()
    fd = FrekiDoc.read(path)
    for line in fd.text_lines():
        text_feats.update(['word_'+w.lower() for w in re.findall('\w+', line)])

    text_dict = dict(text_feats)
    text_dict.update({f:1 for f in url_feats})

    return text_dict

class ClassifierWrapper(object):
    def __init__(self):
        self.learner = LogisticRegression()
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = SelectKBest(chi2, 10000)
        self.classes = []

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

def train_classifier(config):
    """
    Train the classifier

    :type config: ConfigParser
    """


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', help='Alternative config file.', type=ConfigParser.read)

    args = p.parse_args()

    url_mappings = get_urls(defaults.defaults().get('url_list'))
    # find_docs(url_mappings=url_mappings, **defaults.defaults())
    #
    classify_docs(url_mappings=url_mappings, **defaults.defaults())
