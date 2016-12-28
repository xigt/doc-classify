# doc-classify
This is a simple document classifier to be used with `freki` files to determine which documents out of a set are likely linguistic documents.

## Setup
The included `defaults.ini.sample` file contains a number of settings for use with the classifier. These settings include:

* `label_path`: The path for a text document containing a series of rows of the form `{doc_id} 1|0`, where a `1` corresponds to a document of the desired type and `0` is a negative example.
* `url_list`: The path for a text document containing a series of rows of the form `{doc_id} {URL}` where `{URL}` is the URL from which the document was downloaded. This file is optional but reccomended.
* `doc_root`: The location of the root directory where the files are located. It is expected that files will be some number of subdirectories beneath this root, and named as `{doc_id}.txt`.
* `doc_glob`: Underneath the `doc_root`, this 