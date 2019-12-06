You need python 3 to run this script.
# Set up
1. You may want to use venv to create an virtual environment
```bash
    python3 -m venv env
```
2. Activate the virtual environment
```bash
    source ./env/bin/activate
```
3. Install requirement libraries and download required data
```bash
    python setup.py
```

# Train
```
    python train.py
```
for more detail about available arguments, run
```
    python train.py --help
```
train topk model with lemmatizer and stemmer
```
    python train.py -m topk -l -s
```
train topk model without preprocessor
```
    python train.py -m topk 
```

# Predict
```
    python predict.py
```
for more detail about available arguments, run
```
    python predict.py --help
```
predict useing topk model with lemmatizer and stemmer
```
    python predict.py -m topk -l -s
```
predict useing topk model without preprocessor
```
    python predict.py -m topk 
```
predict useing topk model without preprocessor with custom k
```
    python predict.py -m topk -k 5
```
# Evaluate
```
    python evaluate.py input output
```
# Testing
There are plenty of unittests, run
```bash
    python -m unittest
```
