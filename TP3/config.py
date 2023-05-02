# Read from config.json file

import json

def load_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['perceptron']

def ex2_test_size():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['ex2_test_size']

def ex3_test_size():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['ex3_test_size']
