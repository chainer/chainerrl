# Contributing to ChainerRL

Any kind of contribution to ChainerRL would be highly appreciated!

Contribution examples:
- Thumbing up to good issues or pull requests :+1:
- Opening issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
- Sending pull requests

If you could kindly send a PR to ChainerRL, please make sure all the tests successfully pass.

## Testing

To test chainerrl modules, install `nose` and run `nosetests`. Pass `-a '!gpu'` to skip tests that require gpu.

To test examples, run `test_examples.sh [gpu device id]`. `-1` would run examples with only cpu.
