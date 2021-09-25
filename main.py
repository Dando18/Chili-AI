'''
author: Daniel Nichols
date: September 2021
'''
# std imports
from argparse import ArgumentParser

# local imports
from dataset import get_dataset

def get_args():
    ''' Parse CL parameters with argparse
    '''
    parser = ArgumentParser()

    # meta args
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity')

    # input data args
    parser.add_argument('--dataset', required=True, help='path to dataset')
    parser.add_argument('-t', '--tokenize', choices=['word', 'letter', 'step', 'none'],
        default='word', help='How to tokenize input data.')
    parser.add_argument('--batch-size', type=int, default=32, 
        help='training batch size.')
    parser.add_argument('--sequence-length', type=int, help='sequence length of data.')
    
    # markov args
    parser.add_argument('--markov', action='store_true', help='do markov training.')
    parser.add_argument('--markov-num-sentences', type=int, default=5,
        help='Number of "sentences" to generate with markovify.')
    parser.add_argument('--markov-state-size', type=int, default=2, 
        help='how many hidden markov states to use.')

    # lstm args
    parser.add_argument('--lstm', action='store_true', help='do lstm training.')
    parser.add_argument('--lstm-load-model', type=str, help='where to load model')
    parser.add_argument('--lstm-save-model', type=str, help='where to save model')
    parser.add_argument('--lstm-embedding-dim', type=int, default=32, 
        help='size of embedding in lstm network.')
    parser.add_argument('--lstm-hidden-dim', type=int, default=32, 
        help='number of hidden states in lstm network.')
    parser.add_argument('--lstm-layers', type=int, default=2, 
        help='number of layers in lstm network.')
    parser.add_argument('--lstm-dropout', type=float, default=0.2,
        help='dropout in lstm layer.')
    parser.add_argument('--lstm-learning-rate', type=float, default=0.1,
        help='learning rate.')
    parser.add_argument('--lstm-optimizer', choices=['sgd', 'adam'], 
        default='adam', help='optimizer to use in training.')
    parser.add_argument('--lstm-loss', choices=['nll', 'cross_entropy'], 
        default='cross_entropy', help='loss to use in training.')
    parser.add_argument('--lstm-epochs', type=int, default=0,
        help='number of training epochs.')
    parser.add_argument('--lstm-num-sentences', type=int, default=20, 
        help='number of sentences to generate.')

    # generating args
    parser.add_argument('--generate-recipes', nargs='?', type=int, const=1, 
        help='Number of recipes to generate.')

    return parser.parse_args()


def main():
    args = get_args()

    # load data
    dataset = get_dataset(args.dataset, tokenize=args.tokenize, 
        batch_size=args.batch_size, sequence_length=args.sequence_length, 
        verbose=args.verbose)

    if args.markov:
        assert args.tokenize == 'none', 'Markov requires \'none\' tokenizer.'
        import markov
        markov.do_markov_training(dataset, state_size=args.markov_state_size, 
                num_generate=args.generate_recipes,
                num_sentences=args.markov_num_sentences, verbose=args.verbose)

    if args.lstm:
        assert args.tokenize != 'none', 'LSTM requires tokenized input data.'
        import lstm
        ds, vocab = dataset
        lstm.do_lstm_training(ds, vocab, embedding_dim=args.lstm_embedding_dim, 
            hidden_dim=args.lstm_hidden_dim, num_layers=args.lstm_layers,
            dropout=args.lstm_dropout, loss=args.lstm_loss, 
            optim=args.lstm_optimizer, lr=args.lstm_learning_rate, 
            epochs=args.lstm_epochs, save_model=args.lstm_save_model, 
            load_model=args.lstm_load_model, 
            num_sentences=args.lstm_num_sentences,
            num_recipes=args.generate_recipes, verbose=args.verbose)


if __name__ == '__main__':
    main()