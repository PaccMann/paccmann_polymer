from typing import Generator
import json
from collections import OrderedDict
from typing import Any, Union


def load_tsv(filename: str):
    with open(filename, 'r') as f:
        return [x.split('\t') for x in f.readlines()]


def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def load_csv(filename: str):
    with open(filename, 'r') as f:
        return [x for line in f.readlines() for x in line.split(',')]


def nested_get(dictionary, keys):
    """Get nested fields in a dictionary"""
    to_return = dictionary.get(keys[0], None)
    for key in keys[1:]:
        if to_return:
            to_return = to_return.get(key, None)
        else:
            break
    return to_return


def deal_with_numbers(obj):
    """Handle numeric values"""
    if obj is None:
        return None
    if type(obj) is str:
        if not obj.isdigit():
            return None
    keys = list(obj.keys())
    if len(keys) > 1:
        raise IndexError(keys)
    return obj[keys[0]]


def get_id(obj):
    """Get object identifier."""
    return nested_get(obj, ['_id', '$oid'])


def flatten(container: list) -> Generator:
    """Flatten any list of arbitrary nested depth"""
    for i in container:
        if isinstance(i, list):
            for j in flatten(i):
                yield j
        else:
            yield i


def load_mongo_json(filepath: str) -> OrderedDict:
    """Load a JSONL from a file.

    Args:
        filepath (str): Filepat to JSON file.

    Returns:
        dict: Data loaded from JSON file
    """
    with open(filepath) as f:
        raw = f.read()
        first_char = raw[0]
        if first_char == '[':
            _data = json.loads(raw)
        elif first_char == '{':
            # Sometimes the dump files don't star with square brackets and each
            # sample is in a separate line
            _data = [json.loads(line.strip()) for line in raw.split('\n')]
        else:
            raise ValueError(f'Unexpected first char `{first_char}` in file.')
    data = OrderedDict({get_id(obj): obj for obj in _data})
    return data


def check_field_val(key: Any) -> str:
    TYPE_KEYS = ['$numberInt']

    if isinstance(key, str):
        return key
    elif isinstance(key, dict):
        if len(key) > 1:
            raise NotImplementedError(
                'Unsure on how to deal with fields with more than one element'
            )
        type_key, key = list(key.items())[0]
        if type_key not in TYPE_KEYS:
            raise NotImplementedError(
                f'Unsure on how to deal with key type {type_key}'
            )
        return key
    else:
        raise ValueError(f'Expected key of type str or dict, not: {key}')
