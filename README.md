## Deep text matching
Text matching using several deep models.

## Implemented models
- Baseline model: Word vectors averaging + Linear Transformation.
- Deep Bi-LSTMs based model.
- RNMT+ encoder based model.
- Transformer encoder based model.
- Multi head attention based model.

## Get started
- The main entry point is at `train.py`.
- Test with very small data set using `test.py` before training.
- Grid search hyper params with `grid_search.py`.
- Use `apply.py` to load model trained.
