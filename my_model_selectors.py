import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bic = -np.inf
        bic_best = bic
        state_best = 0
        
        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(state)
                log_Likelihood = hmm_model.score(self.X, self.lengths)
                
                # number of data points and features
                data_n, features = self.X.shape
                param_n = state**2 + 2 * state * features - 1
                bic = -2 * log_Likelihood + param_n * np.log(data_n)
                
            except Exception as e:
#                 print(e)
                pass
                
            if bic_best < bic:
                bic_best = bic
                state_best = state

        return self.base_model(state_best)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
#         warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore")

        # TODO implement model selection based on DIC scores
        dic = -np.inf
        dic_best = dic
        state_best = 0
        
        for state in range(self.min_n_components, self.max_n_components + 1):
            # build model
            try:
                # calculate likelihood
                hmm_model = self.base_model(state)
                log_Likelihood = hmm_model.score(self.X, self.lengths)

                # DIC calculation
                word_scores = 0
                # X(all but i)
                words = list(self.words)
                words.remove(self.this_word)
                
                # SUM(log(P(X(all but i))
                for word in words:
                    X, lengths = self.hwords[word]
                    word_scores += hmm_model.score(X, lengths)

                # calculate score
                dic = log_Likelihood - word_scores / len(words)
                
            except Exception as e:
                # print(e)
                pass
                
            if dic_best < dic:
                dic_best = dic
                state_best = state

        return self.base_model(state_best)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        # raise NotImplementedError

        kf = KFold(min(3, len(self.sequences[0])))
        log_Likelihood_dict = {}
        
        # compute range of n (including both numbers)
        for state in range(self.min_n_components, self.max_n_components + 1):
            log_Likelihood = []
            for kf_train_index, kf_test_index in kf.split(self.sequences[0]):
                try:
                    train_x, train_y = combine_sequences(kf_train_index, self.sequences)
                    test_x, test_y = combine_sequences(kf_test_index, self.sequences)
                    
                    hmm_model = self.base_model(state).fit(train_x, train_y)
                    log_Likelihood.append(hmm_model.score(test_x, test_y))
                    log_Likelihood_dict[state] = np.mean(log_Likelihood)

                except Exception as e:
#                     print(e)
                    pass

        # select the best model by choosing max mean likelihood
        if len(log_Likelihood_dict) > 0:
            state_best = max(log_Likelihood_dict, key=log_Likelihood_dict.get)
        else:
            state_best = self.n_constant
        
        # return the mode with best_state of minimal loglikelihood
        return self.base_model(state_best)

        
