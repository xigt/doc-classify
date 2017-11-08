#!/usr/bin/env python
from setuptools import setup

setup(name='doc-classify',
      version='1.0.0',
      description='Document-level classifier for IGT-containing documents, part of RiPLEs pipeline.',
      author='Ryan Georgi',
      author_email='rgeorgi@uw.edu',
      url='https://github.com/xigt/doc-classify',
      scripts=['doc-classify'],
      install_requires = [
          'scikit-learn >= 0.18',
          'numpy',
          'freki',
          'riples_classifier',
      ],
      dependency_links = [
          'https://github.com/xigt/freki/tarball/master#egg=freki-0.1.0',
          'https://github.com/xigt/riples-classifier/tarball/master#egg=riples-classifier-0.1.0',
      ],

      )
