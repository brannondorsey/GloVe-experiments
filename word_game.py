import argparse, utils, sys, readline
from termcolor import colored
from scipy.spatial.distance import cosine

def word_arithmetic(start_word, minus_words, plus_words, word_to_id, id_to_word, df):
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

	return None, result

def find_nearest(skip_words, vec, id_to_word, df, num_results=1, method='cosine'):

	if method == 'cosine':
		minim = [] # min, index
		for i, v in enumerate(df):
			# skip the base word, its usually the closest
			if id_to_word[i] in skip_words:
				continue
			dist = cosine(vec, v)
			minim.append((dist, i, v))
		minim = sorted(minim, key=lambda v: v[0])
		# return list of (word, cosine distance, vector) tuples
		return [(id_to_word[minim[i][1]], minim[i][0], minim[i][2]) for i in range(num_results)]
	else:
		raise Exception('{} is not an excepted method parameter'.format(method))

def eval_expression(expr, word_to_id, id_to_word, df):
	start_word, minus_words, plus_words = parse_expression(expr)
	err, vec = word_arithmetic(start_word=start_word,
		                          minus_words=minus_words,
		                          plus_words=plus_words,
		                          word_to_id=word_to_id,
		                          id_to_word=id_to_word,
		                          df=df)
	if err == None:
		return vec, [start_word] + minus_words + plus_words # vector, skip words
	else:
		raise Exception('Error: {} not found in the dataset.'.format(err))

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

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_dim', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=100,
						help='What vector GloVe vector depth to use (default: 100).')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=10000,
						help='The number of lines to read from the GloVe vector file (default: 10000).')
	parser.add_argument('--soft_score', '-s',
						action='store_true',
						help='points are scored relative to the distance a '
						'player\'s word is from the result of the '
						'input expression. This is in contrast to the default '
						'1 point per-round scoring system. Soft scoring is '
						'recommended for a more fair-and-balanced game experience (default: false)')
	parser.add_argument('--glove_path', '-i',
		                default='data/glove',
		                help='GloVe vector file path (default: data/glove)')
	return parser.parse_args()

def game_setup(args):

	gs = {} # game state
	gs['players'] = read_players()
	gs['winning_score'] = read_winning_score(len(gs['players'].keys()))
	gs['turn_number'] = 0
	return gs

def read_players():
	players = {}
	while len(players.keys()) == 0:
		print('Enter the name of each player, seperated by commas.')
		names = input('> ').split(',')
		confirm = input('There are {} players correct? [yes]: '.format(len(names)))
		if confirm == '' or confirm.lower() == 'yes':
			for name in names:
				players[name.strip()] = 0 # start with a  score of zero
			return players

def read_winning_score(num_players):
	# todo recommend a winning score based on number of players
	winning_score = 0
	while winning_score == 0:
		score = input('What score would you like to play to? [10]: ')
		if score == '':
			winning_score = 10
			return winning_score
		else:
			try:
				winning_score = int(score)
			except ValueError as err:
				print('Invalid score, please try again.')
				break
			return winning_score

def print_standings(gs):
	print()
	standings = ''
	for name, score in gs['players'].items():
		standings += '     {}: {}'.format(name, score)
	print(standings)
	print()

def turn(gs, word_to_id, id_to_word, df, soft_score):

	gs['turn_number'] += 1
	names = list(gs['players'].keys())
	current_player = names[(gs['turn_number'] % len(names) - 1)]
	while True:
		expr = input('{}, please enter a word expression:\n> '.format(current_player))
		try:
			vec, skip_words = eval_expression(expr, word_to_id, word_to_id, df)
		except Exception as err:
			print(err)
			continue
		break

	answers = {}
	for name in gs['players']:
		while True:
			word = input('{}, please enter your answer: '.format(name))
			if word in word_to_id:
				answers[name] = df[word_to_id[word]]
				break
			else:
				print('{} is not in the dataset, please another word.'.format(word))

	answer_word, answer_dist, answer_vec = find_nearest(skip_words, vec, id_to_word, df)[0]
	# transform answers from vectors to distances
	for k, v in answers.items():
		answers[k] = cosine(v, answer_vec)

	winner = min(answers, key=answers.get)

	if not soft_score:
		gs['players'][winner] += 1
	else:
		for name in answers:
			gs['players'][name] += round(answers[name], 2)

	print('Computer says {} = {}'.format(expr, colored(answer_word, 'cyan')))
	print('{} wins this round.'.format(colored(winner, 'green')))
	print_standings(gs)

if __name__ == '__main__':

	args = parse_args()
	vector_file = args.glove_path + '/' + 'glove.6B.' + str(args.vector_dim) + 'd.txt'

	df, labels_array = utils.build_word_vector_matrix(vector_file, args.num_words)
	word_to_id, id_to_word = utils.get_label_dictionaries(labels_array)

	gs = game_setup(args)

	while max(gs['players'].values()) < gs['winning_score']:
		turn(gs, word_to_id, id_to_word, df, args.soft_score)

	print('{} is the winner!'.format(colored(max(gs['players'], key=gs['players'].get), 'green')))
