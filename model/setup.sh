#!/bin/bash
pip install --user numpy
pip install --user pycocotools
pip install --user nltk
python3 -c "import nltk; nltk.download('punkt')"