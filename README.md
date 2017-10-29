# GloVe Experiments

This repository contains a few brief experiments with [Stanford NLP's GloVe](https://nlp.stanford.edu/projects/glove/), an unsupervised learning algorithm for obtaining vector representations for words. Similar to Word2Vec, GloVe creates a continuous N-dimensional representation of a word that is learned from its surrounding context words in a training corpus. Trained on a large corpus of text, these co-occurance statistics (an N-dimensional vector embedding) cause semantically similar words to appear near each-other in their resulting N-dimensional embedding space (e.g. "dog" and "cat" may appear nearby a region of other pet related words in the embedding space because the context words that surround both "dog" and "cat" in the training corpus are similar).

I've created three small python programs for exploring GloVe embeddings:

- `word_arithmetic.py`: Create word analogy searches using basic arithmetic operations (e.g. `king - man + women = queen`).
- `word_game.py`: A small terminal-based multiplayer text game for creating word analogies.
- `word_clustering.py`: Create [K-Means clusters](https://en.wikipedia.org/wiki/K-means_clustering) using GloVe embeddings. Saves results to JSON.

All three scripts use the GloVe.6B pre-trained word embeddings created from the combined Wikipedia 2014 and Gigaword 5 datasets. They were trained using 6 billion tokens and contains 400,000 unique lowercase words. Trained embeddings are provided in 50, 100, 200, and 300 dimensions (822 MB download).

## Getting Started

These small experiments can be run in MacOS or Linux environments (sorry ~~not sorry~~ Windoze users).

```bash
# clone this repo
git clone https://github.com/brannondorsey/GloVe-experiments.git
cd GloVe-experiments

# install python dependencies
pip3 install -r requirements.txt

# dowload the pre-trained embeddings. This might take a while...
./download_data.sh
```

## Word Arithmetic

`word_arithmetic.py` allows you to write simple +/- arithmetic operations using words to find the closest approximated resulting word from the given word expression. Math operations are applied in the embedding space and a K-nearest-neighbor search is used to display the `K` words closest to the result of the algebraic transformation.

```bash
python3 word_arithmetic.py
> king - man + woman

queen                0.22
```

`word - word + word` is the traditional word analogy format, however `word_arithmetic.py` supports any number of `+` or `-` operations provided all words are in the database. The meaning of less traditional expressions, `word + word + word...` is more ambiguous but can lead to interesting results nonetheless. Specifying an order of operations is not supported at this time (e.g. `(word - word) + word`).

By default, `word_arithmetic.py` loads the 10,000 most frequently used words from the dataset and uses a 100-dimensional embedding vector. It also prints only the single nearest word to the resulting vector point from the expression (the "nearest neighbor"). You can specify your own values for each of these parameters if you would like:

```bash
python3 word_arithmetic.py --num_words 100000 --vector_dim 300 --num_output 10
> king - man + woman

queen                0.31
monarch              0.44
throne               0.44
princess             0.45
mother               0.49
daughter             0.49
kingdom              0.50
prince               0.50
elizabeth            0.51
wife                 0.52
```

Increasing `--num_words` and `--vector_dim` increases the number of usable words in the dictionary and accuracy of the resulting word expressions respectively. Increasing either will increase the processing time for each expression as well as the memory requirements needed to run the program.

```
usage: word_arithmetic.py [-h] [--vector_dim {50,100,200,300}]
                          [--num_words NUM_WORDS] [--num_output NUM_OUTPUT]
                          [--glove_path GLOVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --vector_dim {50,100,200,300}, -d {50,100,200,300}
                        What vector GloVe vector depth to use (default: 100).
  --num_words NUM_WORDS, -n NUM_WORDS
                        The number of lines to read from the GloVe vector file
                        (default: 10000).
  --num_output NUM_OUTPUT, -o NUM_OUTPUT
                        The number of result words to display (default: 1)
  --glove_path GLOVE_PATH, -i GLOVE_PATH
                        GloVe vector file path
```

## Word Game

`word_game.py` is a small text-based multiplayer game where players take turns creating and answering `word_arithmetic.py`-style word expressions. Players win points when they propose a solution word to a word expression that is nearest to the answer word out of all players guesses.

```
Enter the name of each player, seperated by commas.
> bob, alice
There are 2 players correct? [yes]: yes
What score would you like to play to? [10]: 10
alice, please enter a word expression:
> home - earth + space
alice, please enter your answer: rocket
bob, please enter your answer: moon
Computer says home - earth + space = office
bob wins this round.

     alice: 0     bob: 1

bob, please enter a word expression:
>
```

The game is far from perfect, and the automated judging can be aggravating at times (try with `--soft_score`), but it can lead to some fun times given the right crowd 💻🍻🎉. Increase the dictionary size and vector dimensions for best results:

```bash
python3 word_game.py --vector_dim 200 --num_words 100000 --soft_score
```

```
usage: word_game.py [-h] [--vector_dim {50,100,200,300}]
                    [--num_words NUM_WORDS] [--soft_score]
                    [--glove_path GLOVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --vector_dim {50,100,200,300}, -d {50,100,200,300}
                        What vector GloVe vector depth to use (default: 100).
  --num_words NUM_WORDS, -n NUM_WORDS
                        The number of lines to read from the GloVe vector file
                        (default: 10000).
  --soft_score, -s      points are scored relative to the distance a player's
                        word is from the result of the input expression. This
                        is in contrast to the default 1 point per-round
                        scoring system. Soft scoring is recommended for a more
                        fair-and-balanced game experience (default: false)
  --glove_path GLOVE_PATH, -i GLOVE_PATH
                        GloVe vector file path (default: data/glove)
```

## Word Clustering

`word_clustering.py` uses unsupervised learning to clusters words into related groups using K-Means.

```bash
python3 word_clustering.py
No cached cluster found. Clustering using K-Means...
Saved 1000 clusters to data/cache/100D_10000-words_1000-clusters.json. Cached for later use.
CLUSTER 1: athens, stockholm, oslo, helsinki
CLUSTER 2: long, short, longer, normal, usual, periods, lengthy, shorter, duration
CLUSTER 3: current, term, future, key, position, primary, internal, existing, core, external
CLUSTER 4: newton, luther, canon
CLUSTER 5: ball, pitch, catch, throw, balls, swing, bat, kicked, opener, slip, spell, foul, knock, pitches, toss, kicking, bounced, kicks, scoreboard, bounce
CLUSTER 6: popular, famous, prominent, notable, influential, renowned, well-known, famed, acclaimed, finest
CLUSTER 7: affected, affect, affecting, affects
CLUSTER 8: assassination, murdered, slain, assassinated
CLUSTER 9: jordan, carter, jimmy
CLUSTER 10: 1999, 1994, 1995, 1993, 1992, 1991, 1990, 1989, 1988, 1986, 1987, 1984, 1980, 1985, 1979, 1983, 1982, 1981
CLUSTER 11: alongside, joining, touring, completing, toured, thereafter, whilst, filming, assignment, boarding, stint
CLUSTER 12: 10, 20, 15, 30, 11, 12, 18, 25, 14, 13, 16, 17, 24, 19, 22, 21, 23, 26, 28, 27, 31, 29
CLUSTER 13: support, provide, aid, access, provided, additional, offers, relief, provides, assistance, providing, funding
CLUSTER 14: communist, regime, dictator, suharto, dictatorship, communism, monarchy
...
--- 28.54 seconds ---
```

Clusters are printed to the screen and also saved as JSON arrays in `data/cache`. By default, the script clusters the 10,000 most-common words from GloVe.6B into 1,000 clusters using 100-D vector embeddings. This can be changed like so:

```bash
# note: this will take a *long* time to run...
python3 word_clustering.py --num_words 100000 --num_clusters 10000 --vector_dim 300
```

```
usage: word_clustering.py [-h] [--vector_dim {50,100,200,300}]
                          [--num_words NUM_WORDS]
                          [--num_clusters NUM_CLUSTERS] [--n_jobs N_JOBS]
                          [--glove_path GLOVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --vector_dim {50,100,200,300}, -d {50,100,200,300}
                        What vector GloVe vector dimension to use (default:
                        100).
  --num_words NUM_WORDS, -n NUM_WORDS
                        The number of lines to read from the GloVe vector file
                        (default: 10000).
  --num_clusters NUM_CLUSTERS, -k NUM_CLUSTERS
                        Number of resulting word clusters. The number of K in
                        K-Means (default: 1000).
  --n_jobs N_JOBS, -j N_JOBS
                        Number of cores to use when fitting K-Means. -1 = all
                        cores. More cores = less time, more memory (default:
                        -1).
  --glove_path GLOVE_PATH, -i GLOVE_PATH
                        GloVe vector file path (default: data/glove)
```

## License and Attribution

All code is released under an [MIT license](LICENSE). You are free to copy, edit, share, or sell it under those terms.

### GloVe citation

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).

```
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}
```
