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

# Predict
```
    python predict.py
```
for more detail about available arguments, run
```
    python predict.py --help
```

# Evaluate
```
    python evaluate input output
```
# Testing
There are plenty of unittests, run
```bash
    python -m unittest
```
