# Emotional Relationship Classification

The repository contains the code for the classification of emotional character relationships with GRU (Keras). Current implementation is based on the following paper:

> Kim, E., & Klinger, R. (2019, June). Frowning Frodo, Wincing Leia, and a Seriously Great Friendship: Learning to Classify Emotional Relationships of Fictional Characters. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 647-653).

## Dependencies

```bash
keras=2.4.3
tensorflow=2.3.1
sklearn=0.23.2
```

The implementation requires you to download [`glove.6B.300d`](http://nlp.stanford.edu/data/glove.6B.zip) embeddings. Place the unzipped `txt` file with embeddings to the `embeddings` directory.

`fanfic-corpus.txt` is the file with data. Splitting data into train, dev, and test is handled by utility functions.

## Run the model

`python model.py <number-of-classes> <indicator-type> <number-of-epochs> <directed-undirected> <window-size>`

Make sure you pass all arguments.

Possible values for the arguments:
`number-of-classes`: `8class`, `5class`, `2class`(see paper for details)
`indicator-type`: `mrole`,`no-ind`, `role`, `mentity`, `entity` (see paper for details)
`number-of-epochs`: any `int`
`directed-undirected`: `directed` or `undirected` (see paper for details)
`window-size`: any `int` from 1 to 20

