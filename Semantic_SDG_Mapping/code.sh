#!/bin/bash

sudo chmod 775 input.txt
# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_string>"
    exit 1
fi

# Get the input string
input_string="$1"

# Save the input as input.txt in the current directory
sudo echo "$input_string" > input.txt

# Run the Semantic_SDG_Mapping.py script
# Install pip if not already installed
#sudo python3.10 -m pip --version >/dev/null 2>&1 || sudo apt-get install python3-pip -y

# Install required Python libraries
#sudo python3.10 -m pip install --upgrade pip
#sudo python3.10 -m pip install pandas==2.2.2 numexpr==2.10.0 bottleneck==1.3.8 numpy==1.24.0 nltk==3.8.1 scikit-learn==1.4.2 scipy==1.8.0 gensim==4.3.2

# Download NLTK resources
#sudo python3.10 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
#sudo python3 test.py

