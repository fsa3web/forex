import json
import os

CURRENCY_PAIRS_FILE = 'currency_pairs.json'

def load_currency_pairs():
    if os.path.exists(CURRENCY_PAIRS_FILE):
        with open(CURRENCY_PAIRS_FILE, 'r') as f:
            return json.load(f)
    return ["AUDCAD", "EURUSD", "AUDJPY", "NZDUSD", "AUDNZD"]  # Default pairs

def save_currency_pairs(pairs):
    with open(CURRENCY_PAIRS_FILE, 'w') as f:
        json.dump(pairs, f)

def add_currency_pair(pair):
    pairs = load_currency_pairs()
    if pair not in pairs:
        pairs.append(pair)
        save_currency_pairs(pairs)
        return True
    return False

def remove_currency_pair(pair):
    pairs = load_currency_pairs()
    if pair in pairs:
        pairs.remove(pair)
        save_currency_pairs(pairs)
        return True
    return False

def get_currency_pairs():
    return load_currency_pairs()
