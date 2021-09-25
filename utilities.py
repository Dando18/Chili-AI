'''
author: Daniel Nichols
date: September 2021
'''

def vprint(verbose, message, **kwargs):
    ''' Verbose print utility.
    '''
    if verbose:
        print(message, **kwargs)