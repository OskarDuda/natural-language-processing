# Natural language processing
## Installation
```bash
git clone https://github.com/OskarDuda/natural-language-processing.git
cd natural-language-processing
```

creating a Python virtual environment:
```bash
virtualenv venv -p python3
source venv/bin/activate
```

intalling dependencies:
```bash
pip3 install -r requirements.txt
```

## How to run it
If virtual environment has been created:
```bash
source venv/bin/activate
```

running the script:
```bash
python3 gender_prediction_by_tweets.py 
```

the resulting score of the prediction will be printed in the console. 
Current version doesn't support model selection.