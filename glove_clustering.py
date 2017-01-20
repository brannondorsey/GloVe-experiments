# Notes for extension of script:
# 	- User readline() to interactively search for word groups
# 	- On a word miss, use L2 or cosine distance to select the nearest word vector
# 		- This would require all 6B tokens to loaded in ram (but not clustered)
#		- Or use levenshtein distance assuming the word is spelled the same.
#   - Provide an interface to perform basic arithmetic on words (king - man + woman = queen) 
# Look at this result from 2014 English Wikipedia: 
# 'islamic', 'militant', 'islam', 'radical', 'extremists', 'islamist', 'extremist', 'outlawed'
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
from scipy.spatial.distance import cosine
import numpy as np
import os, sys, codecs, argparse, pprint, json, time

'''Serializable/Pickleable class to replicate the functionality of collections.defaultdict'''
class autovivify_list(dict):
        def __missing__(self, key):
                value = self[key] = []
                return value

        def __add__(self, x):
                '''Override addition for numeric types when self is empty'''
                if not self and isinstance(x, Number):
                        return x
                raise ValueError

        def __sub__(self, x):
                '''Also provide subtraction method'''
                if not self and isinstance(x, Number):
                        return -1 * x
                raise ValueError

def build_word_vector_matrix(vector_file, n_words):
	'''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
	np_arrays = []
	labels_array = []

	with codecs.open(vector_file, 'r', 'utf-8') as f:
		for i, line in enumerate(f):
			sr = line.split()
			labels_array.append(sr[0])
			np_arrays.append(np.array([float(j) for j in sr[1:]]))
			if i == n_words - 1:
				return np.array(np_arrays), labels_array

def find_word_clusters(labels_array, cluster_labels):
	cluster_to_words = autovivify_list()
	for c, i in enumerate(cluster_labels):
		cluster_to_words[i].append(labels_array[c])
	return cluster_to_words

def get_filename_from_args(args):
	a = (args.vector_depth, args.num_words, args.reduction_factor, int(args.num_words * args.reduction_factor))
	return '{}-d_{}-words_{}-reduction_{}-clusters.json'.format(*a)

def word_arithmetic(base_word, minus_word, plus_word, word_to_id, id_to_word, df):
	'''Returns a word string that is the result of the vector arithmetic'''
	try:
		base_vec  = df[word_to_id[base_word]]
		minus_vec = df[word_to_id[minus_word]]
		plus_vec  = df[word_to_id[plus_word]]
	except KeyError as err:
		print(err)
		return None

	result = base_vec - minus_vec + plus_vec
	words = (base_word, minus_word, plus_word)
	return find_nearest(words, result, id_to_word, df)

def find_nearest(words, vec, id_to_word, df, method='cosine'):

	if method == 'cosine':
		minim = (sys.maxsize, 0) # min, index		
		for i, v in enumerate(df):
			# skip the base word, its usually the closest
			if id_to_word[i] in words:
				continue
			dist = cosine(vec, v)
			if dist < minim[0]:
				minim = (dist, i)
		return id_to_word[minim[1]], minim[0] # word, cosine distance
	else:
		raise Exception('{} is not an excepted method parameter'.format(method))

def save_json(filename, results):
	with open(filename, 'w') as f:
		print(results)
		json.dump(results, f)

def load_json(filename):
	with open(filename, 'r') as f:
		return json.load(f)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_depth', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=50,
						help='What vector GloVe vector depth to use.')
	parser.add_argument('--glove_path', '-i',
		                default='data',
		                help='GloVe vector file path')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=10000,
						help='The number of lines to read from the GloVe vector file.')
	parser.add_argument('--reduction_factor', '-r',
						default=0.1,
						type=float,
						help='Ratio used to determine number of clusters. Higher ratio = more clusters. Num clusters = -n * -r.')
	parser.add_argument('--n_jobs', '-j',
						type=int,
						default=-1,
						help='Number of cores to use when fitting KMeans. -1 = all cores. More cores = less time, more memory.')
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()

	filename = path = '../data/glove_models/{}'.format(get_filename_from_args(args))
	cluster_to_words = None
	start_time = time.time()

	vector_file = args.glove_path + '/' + 'glove.6B.' + str(args.vector_depth) + 'd.txt'
	k = int(args.num_words * args.reduction_factor)
	df, labels_array = build_word_vector_matrix(vector_file, args.num_words)

	# if these are clustering parameters we've never seen before
	if not os.path.isfile(filename):
		
		kmeans_model = KMeans(init='k-means++', n_clusters=k, n_jobs=args.n_jobs, n_init=10)
		kmeans_model.fit(df)

		cluster_labels   = kmeans_model.labels_
		# cluster_inertia = kmeans_model.inertia_
		cluster_to_words = list(find_word_clusters(labels_array, cluster_labels).values())

		# cache these clustering results
		save_json(path, cluster_to_words)
		print('Saved {} clusters to {}'.format(len(cluster_to_words), path))
	
	# if this kmeans fitting has already been cached
	else:

		cluster_to_words = load_json(filename)

	id_to_word = dict(zip(range(len(labels_array)), labels_array))
	word_to_id = dict((v,k) for k,v in id_to_word.items())

	result = word_arithmetic('king', 'man', 'woman', word_to_id, id_to_word, df)
	print(result)

	# for i, words in enumerate(cluster_to_words):
	# 	# print(words)
	# 	pass

	if start_time != None:
			print("--- %s seconds ---" % (time.time() - start_time))
	