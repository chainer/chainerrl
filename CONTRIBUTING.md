# Contributing to ChainerRL

Any kind of contribution to ChainerRL would be highly appreciated!

Contribution examples:
- Thumbing up to good issues or pull requests :+1:
- Opening issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
- Sending pull requests

If you could kindly send a PR to ChainerRL, please make sure all the tests successfully pass.

## Testing

To test chainerrl modules, install and run `pytest`. Pass `-m "not gpu"` to skip tests that require gpu. E.g.
```
$ pip install pytest
$ pytest -m "not gpu"
```


To test examples, run `test_examples.sh [gpu device id]`. `-1` would run examples with only cpu.

## Coding style

We use PEP8. To check your code, use `autopep8` and `flake8` packages.
```
$ pip install autopep8 flake8
$ autopep8 --diff path/to/your/code.py
$ flake8 path/to/your/code.py
```
