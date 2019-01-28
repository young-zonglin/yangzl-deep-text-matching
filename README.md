## Deep text matching
Text matching using several deep models.

## Implemented models
- Baseline model: Word vectors averaging + Fully connected feed-forward layers.
- Deep Bi-LSTMs based model.
- RNMT+ encoder based model.
- Transformer encoder based model.
- Multi head attention based model.

## Get started
- The main entry point is at `train.py`.
- Test with very small dataset using `test.py` before training.
- Grid search hyper-parameters with `grid_search.py`.
- Use `apply.py` to load trained model.
