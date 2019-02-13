# Contributing to Keras-RL

New contributors are very welcomed! If you're interested, please post a message on the [Gitter](https://gitter.im/keras-rl/Lobby).

Here is a list of ways you can contribute to this repository:
- Tackle an open issue on [Github](https://github.com/keras-rl/keras-rl/issues)
- Improve documentation
- Improve test coverage
- Add examples
- Implement new algorithms on Keras-RL (please get in touch on Gitter)
- Link to your personal projects built on top of Keras-RL


## How to run the tests

To run the tests locally, you'll first have to install the following dependencies:
```bash
pip install pytest pytest-xdist pep8 pytest-pep8 pytest-cov python-coveralls
```
You can then run all tests using this command:
```bash
py.test tests/.
```
If you want to check if the files conform to the PEP8 style guidelines, run the following command:
```bash
py.test --pep8
```
