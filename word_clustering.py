# Notes for extension of script:
# 	- User readline() to interactively search for word groups
# 	- On a word miss, use L2 or cosine distance to select the nearest word vector
# 		- This would require all 6B tokens to loaded in ram (but not clustered)
#		- Or use levenshtein distance assuming the word is spelled the same.
#   - Provide an interface to perform basic arithmetic on words (king - man + woman = queen)
# Look at this result from 2014 English Wikipedia:
# 'islamic', 'militant', 'islam', 'radical', 'extremists', 'islamist', 'extremist', 'outlawed'
# 'war' - 'violence' + 'peace' = 'treaty' | 300d

from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import numpy as np
import os, sys, codecs, argparse, pprint, time
from utils import *
from word_arithmetic import *

def find_word_clusters(labels_array, cluster_labels):
	cluster_to_words = autovivify_list()
	for c, i in enumerate(cluster_labels):
		cluster_to_words[i].append(labels_array[c])
	return cluster_to_words

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_dim', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=100,
						help='What vector GloVe vector dimension to use '
							 '(default: 100).')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=10000,
						help='The number of lines to read from the GloVe '
							 'vector file (default: 10000).')
	parser.add_argument('--num_clusters', '-k',
						default=1000,
						type=int,
						help='Number of resulting word clusters. '
						'The number of K in K-Means (default: 1000).')
	parser.add_argument('--n_jobs', '-j',
						type=int,
						default=-1,
						help='Number of cores to use when fitting K-Means. '
						     '-1 = all cores. '
							 'More cores = less time, more memory (default: -1).')
	parser.add_argument('--glove_path', '-i',
		                default='data/glove',
		                help='GloVe vector file path (default: data/glove)')
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()

	filename = path = 'data/cache/{}'.format(get_cache_filename_from_args(args))
	cluster_to_words = None
	start_time = time.time()

	vector_file = args.glove_path + '/' + 'glove.6B.' + str(args.vector_dim) + 'd.txt'
	df, labels_array = build_word_vector_matrix(vector_file, args.num_words)

	# if these are clustering parameters we've never seen before
	if not os.path.isfile(filename):

		print('No cached cluster found. Clustering using K-Means... ')
		kmeans_model = KMeans(init='k-means++', n_clusters=args.num_clusters, n_jobs=args.n_jobs, n_init=10)
		kmeans_model.fit(df)

		cluster_labels   = kmeans_model.labels_
		# cluster_inertia = kmeans_model.inertia_
		cluster_to_words = list(find_word_clusters(labels_array, cluster_labels).values())

		# cache these clustering results
		save_json(path, cluster_to_words)
		print('Saved {} clusters to {}. Cached for later use.'.format(len(cluster_to_words), path))

	# if this kmeans fitting has already been cached
	else:
		print('Cached K-Means cluster found, loading from disk.')
		cluster_to_words = load_json(filename)

	for i, words in enumerate(cluster_to_words):
		print('CLUSTER {}: {}'.format(i + 1, ', '.join(words)))

	if start_time != None:
			print("--- {:.2f} seconds ---".format((time.time() - start_time)))
