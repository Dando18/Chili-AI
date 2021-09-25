'''
author: Daniel Nichols
date: September 2021
'''
# 3rd party imports
import markovify

# local imports
from utilities import vprint


def do_markov_training(dataset, state_size=2, num_generate=1, num_sentences=5, verbose=False):
    ''' Train the markov model using Markovify.
        Args:
            dataset: input dataset of recipes.
            state_size: how many hidden markov states to use.
            num_generate: how many recipes to generate.
            num_sentences: how many lines per recipe to generate.
            verbose: verbosity.
    '''
    SKIP_NONE_CAP = 25

    corpus = '\n<END_STEP>\n'.join(map(lambda x: x['data'], dataset))
    
    model = markovify.NewlineText(corpus, state_size=state_size, well_formed=False)

    for _ in range(num_generate):
        print('Recipe: ')
        for _ in range(num_sentences):
            sent = model.make_sentence(tries=SKIP_NONE_CAP)
            print('\t{}'.format(sent))
        
        print()