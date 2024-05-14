#!/bin/bash

sudo apt update

# Install pip if not already installed
python3.10 -m pip --version >/dev/null 2>&1 || sudo apt-get install python3-pip -y

# Install required Python libraries
python3.10 -m pip install --upgrade pip
python3.10 -m pip install pandas==2.2.2 numexpr==2.10.0 bottleneck==1.3.8 numpy==1.24.0 nltk==3.8.1 scikit-learn==1.4.2 scipy==1.8.0 gensim==4.3.2

# Download NLTK resources
python3.10 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"


# Make the Python file executable
chmod +x Semantic_SDG_Mapping.py

