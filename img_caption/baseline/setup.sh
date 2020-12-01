#!/bin/bash
pip install --user torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user numpy
pip install --user pycocotools
pip install --user nltk
python3 -c "import nltk; nltk.download('punkt')"