# Read from config.json file

import json

def load_kohonen_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['kohonen']