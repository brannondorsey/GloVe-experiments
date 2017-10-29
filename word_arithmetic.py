import argparse, utils, sys, readline
from scipy.spatial.distance import cosine

def word_arithmetic(start_word, minus_words, plus_words, word_to_id, id_to_word, df, num_results=5):
	'''Returns a word string that is the result of the vector arithmetic'''
	try:
		start_vec  = df[word_to_id[start_word]]
		minus_vecs = [df[word_to_id[minus_word]] for minus_word in minus_words]
		plus_vecs  = [df[word_to_id[plus_word]] for plus_word in plus_words]
	except KeyError as err:
		return err, None

	result = start_vec

	if minus_vecs:
		for i, vec in enumerate(minus_vecs):
			result = result - vec

	if plus_vecs:
		for i, vec in enumerate(plus_vecs):
			result = result + vec

	# result = start_vec - minus_vec + plus_vec
	words = [start_word] + minus_words + plus_words
	return None, find_nearest(words, result, id_to_word, df, num_results)

def find_nearest(words, vec, id_to_word, df, num_results, method='cosine'):

	if method == 'cosine':
		minim = [] # min, index
		for i, v in enumerate(df):
			# skip the base word, its usually the closest
			if id_to_word[i] in words:
				continue
			dist = cosine(vec, v)
			minim.append((dist, i))
		minim = sorted(minim, key=lambda v: v[0])
		# return list of (word, cosine distance) tuples
		return [(id_to_word[minim[i][1]], minim[i][0]) for i in range(num_results)]
	else:
		raise Exception('{} is not an excepted method parameter'.format(method))

def parse_expression(expr):

	split = expr.split()
	start_word = split[0]
	minus_words, plus_words = [], []
	for i, token in enumerate(split[1:]):
		if token == '+':
			plus_words.append(split[i + 2])
		elif token == '-':
			minus_words.append(split[i + 2])
	return start_word, minus_words, plus_words

def process(num_results):
	inpt = input('> ')
	if inpt == 'exit':
		exit()
	start_word, minus_words, plus_words = parse_expression(inpt)
	err, results = word_arithmetic(start_word=start_word,
		                          minus_words=minus_words,
		                          plus_words=plus_words,
		                          word_to_id=word_to_id,
		                          id_to_word=id_to_word,
		                          df=df,
								  num_results=num_results)
	if results:
		print()
		for res in results:
			print(res[0].ljust(15), '     {0:.2f}'.format(res[1]))
		print()
	else:
		print('{} not found in the dataset.'.format(err), file=sys.stderr)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_dim', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=100,
						help='What vector GloVe vector depth to use '
						     '(default: 100).')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=10000,
						help='The number of lines to read from the GloVe '
						     'vector file (default: 10000).')
	parser.add_argument('--num_output', '-o',
						type=int,
						default=1,
						help='The number of result words to display (default: 1)')
	parser.add_argument('--glove_path', '-i',
		                default='data/glove',
		                help='GloVe vector file path (default: data/glove)')
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()
	vector_file = args.glove_path + '/' + 'glove.6B.' + str(args.vector_dim) + 'd.txt'

	if args.num_words > 400000:
		print('--num_words must be equal to or less than 400,000. Exiting.')
		exit(1)

	df, labels_array = utils.build_word_vector_matrix(vector_file, args.num_words)
	word_to_id, id_to_word = utils.get_label_dictionaries(labels_array)

	while True:
		process(args.num_output)
