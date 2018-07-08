import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
#     raise NotImplementedError
    
    for X, lengths in test_set.get_all_Xlengths().values():
        # create dict for probabilities
        dict_for_prob = {}
        p_dict = {}
        log_Likelihood_best = -np.inf
        word_best = None
        
        # get model
        for _, (word, hmm_model) in enumerate(models.items()):
            if hmm_model:
                try:
                    # compute loglikelihood
                    log_Likelihood = hmm_model.score(X, lengths)
                    p_dict[word] = log_Likelihood
                    # find the best within this word
                    if log_Likelihood > log_Likelihood_best:
                            log_Likelihood_best = log_Likelihood
                            word_best = word
                            
                except Exception as e:
#                     print(e)
                    pass

        probabilities.append(p_dict)
        guesses.append(word_best)

    return probabilities, guesses