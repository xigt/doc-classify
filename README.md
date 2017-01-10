# doc-classify
This is a simple document classifier to be used with `freki` files to determine which documents out of a set are likely linguistic documents that may contain IGT instances suitable for use with the [IGT classifier](https://github.com/xigt/igtdetect).

## Dependencies
* [`freki`](https://github.com/xigt/freki)
* [`numpy`](http://www.numpy.org/)
* [`scikit-learn`](http://scikit-learn.org/)

## Running the Classifier

Assuming that a configuration file has been appropriately specified according to the Setup section below, the program can be called with one of the three following options:

    ./doc_classify.py train [-c experiment.ini]
    ./doc_classify.py test  [-c experiment.ini]
    ./doc_classify.py nfold [-c experiment.ini]
    
### `train`
The `train` subcommand has the following usage:

	usage: doc_classify.py train [-h] [-v] [-c CONFIG] [-d] [-f FORCE]
	                             [-m MODEL_PATH] [--num-features NUM_FEATURES]
	
	optional arguments:
	  -h, --help                  show this help message and exit
	  -v, --verbose               Increase verbosity of output
	  -c CONFIG, 
	    --config CONFIG           Alternate config file
	  -d, --debug                 Turn on debugging output
	  -f FORCE, 
	    --force FORCE             Overwrite files.
	  -m MODEL_PATH, 
	    --model-path MODEL_PATH   Path to classifier model.
	                        
	  --num-features NUM_FEATURES Number of features to limit
	                              training the model with.


### `test`
The `test` subcommand has the following usage:

	usage: doc_classify.py test [-h] [-v] [-c CONFIG] [-d] [-f FORCE]
	                            [-m MODEL_PATH]
	
	optional arguments:
	  -h, --help                  show this help message and exit
	  -v, --verbose               Increase verbosity of output
	  -c CONFIG, 
	    --config CONFIG           Alternate config file
	  -d, --debug                 Turn on debugging output
	  -f FORCE, 
	    --force FORCE             Overwrite files.
	  -m MODEL_PATH, 
	    --model-path MODEL_PATH   Path to classifier model.
	    
### `nfold`
The `nfold` subcommand has the following usage:

	usage: doc_classify.py nfold [-h] [-v] [-c CONFIG] [-d] [-f FORCE]
	                             [-m MODEL_PATH] [--train-ratio TRAIN_RATIO]
	                             [--nfold-iterations NFOLD_ITERATIONS]
	                             [--random-seed RANDOM_SEED]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -v, --verbose         Increase verbosity of output.
	  -c CONFIG, --config CONFIG
	                        Alternate config
	  -d, --debug           Turn on debugging output
	  -f FORCE, --force FORCE
	                        Overwrite files.
	  -m MODEL_PATH, --model-path MODEL_PATH
	                        Path to classifier model.
	  --train-ratio TRAIN_RATIO
	                        Ratio of training data to testing data, as a decimal
	                        between (0-1].
	  --nfold-iterations NFOLD_ITERATIONS
	                        Number of train/test iterations to perform.
	  --random-seed RANDOM_SEED
	                        Integer to fix randomization of data shuffling between
	                        program invocations.    
	                        
## Setup
The code uses configuration files for most settings of the classifier. Copying the included `defaults.ini.sample` to `defaults.ini` will allow for a number of installation-specific variables to be set, such as required paths to add to the `PYTHONPATH` variable, or the location of training or testing data.

Any option specified in `defaults.ini` may be overridden by using available commandline options with the same name (e.g. `--model-path`) or specifying a new configuration file using `-c` or `--config`.

### Configuration Options

The full list of configuration options is as follows:

#### files and paths

* `label_path`
	* Path at which the labels for training documents will be specified. Our data can be found in the [`doc-classify-data`](https://github.com/xigt/doc-classify-data) repo.

* `model_path`
	* Path at which to save (for training) or load (for testing) the classification model.
* `python_paths`
	* If it is not possible or preferential not to install the dependencies in the usual python installation path, this colon-separated list of paths can be used to add directories containing the required dependencies to the `PYTHONPATH` variable at runtime.
* `train_dirs`
	* Comma-separated list of directories containing training documents (ending with `.freki` or `.freki.gz`.
* `test_dirs`
	* Comma-separated list of directories containing documents to be classified (ending with `.freki` or `.freki.gz`
* `url_list`
	* List of URLs from which documents were retrieved from. Used to extract URL-related features.

#### feature selection
	
* `num_features`
	* The number of features used to train the model may be limited to the `n` number of features specified here.

#### n-fold testing options
	
* `train_ratio`
	* The ratio, expressed as a decimal between `(0,1]`of training data:testing data, to be used for each iteration of training.
* `nfold_iterations`
	* The number of training/testing partitions to perform.
* `random_seed`
	* An integer that may be changed as desired to change the random shuffling of data in the nfold test, while retaining reproducability between program invocations.