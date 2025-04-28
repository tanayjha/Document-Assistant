## Step 1: Install python

Run the following to see if one of them produces an output to confirm if python is already installed:
`python --version` or `python3 --version`

If they say command not found, install python using the following steps:
(Any 3.10+ version should work but I have tested using 3.10.7 hence installing that)

1. Download the macOS installer for Python 3.10.7: https://www.python.org/ftp/python/3.10.7/python-3.10.7-macos11.pkg
2. Run the installer and follow the prompts.
3. Verify installation: `python3 --version`

### Step 2: Install Ollama

First, download and install Ollama from the official website: https://ollama.com/download/.

Once the models are installed, run the following command in terminal:

`ollama run llama3.2`

## Step 3: Clone the code

## Step 4: Create virtual environment

### On Linux/Mac:
```
python3.11 -m venv myvenv
source myvenv/bin/activate
```

### On Windows:

```
python3.11 -m venv myvenv
myvenv\Scripts\activate
```

### Step 5: Install dependencies

`pip install -r requirements.txt`

### Step 6: Run the code

From the main folder run the following command:
`streamlit run main.py`

This should open up the app in: http://localhost:8501/