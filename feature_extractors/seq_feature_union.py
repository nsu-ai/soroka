import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition


class SeqFeatureUnion(_BaseComposition, TransformerMixin):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list
        self._validate_transformers()

    def get_params(self, deep=True):
        return self._get_params('transformer_list', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('transformer_list', **kwargs)
        return self

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)
        self._validate_names(names)
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
            hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform. '%s' (type %s) doesn't" %
                                (t, type(t)))

    def _iter(self):
        return ((name, trans) for name, trans in self.transformer_list if trans is not None)

    def fit(self, X, y=None):
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = [trans.fit(X, y) for _, trans in self._iter()]
        self._update_transformer_list(transformers)
        return self

    def transform(self, X):
        EPS = 1e-6
        X_trans = self.transformer_list[0][1].transform(X)
        self._validate_X(X_trans, 0)
        n_texts = X_trans.shape[0]
        n_tokens = X_trans.shape[2]
        embedding_size = X_trans.shape[3] - 1
        for text_ind in range(n_texts):
            for token_ind in range(n_tokens):
                mask = X_trans[text_ind][0][token_ind][embedding_size]
                for feature_ind in range(embedding_size):
                    X_trans[text_ind][0][token_ind][embedding_size - feature_ind] = \
                        X_trans[text_ind][0][token_ind][embedding_size - feature_ind - 1]
                X_trans[text_ind][0][token_ind][0] = mask
        for transformer_ind in range(1, len(self.transformer_list)):
            transformer_name = self.transformer_list[transformer_ind][0]
            X_trans_ = self.transformer_list[transformer_ind][1].transform(X)
            self._validate_X(X_trans_, transformer_ind)
            if X_trans.shape[0:3] != X_trans_.shape[0:3]:
                raise ValueError('Outputs of transformer "{0}" do not correspond to outputs of previous transformers. '
                                 '{1} != {2}.'.format(transformer_name, X_trans_.shape[0:3], X_trans.shape[0:3]))
            for text_ind in range(n_texts):
                for token_ind in range(n_tokens):
                    if abs(X_trans[text_ind][0][token_ind][0] - X_trans_[text_ind][0][token_ind][-1]) > EPS:
                        raise ValueError('Outputs of transformer "{0}" do not correspond to outputs of previous '
                                         'transformers. Masks for (text {1}, token {2}) are not equal.'.format(
                            transformer_name, text_ind, token_ind))
            X_trans = np.concatenate((X_trans, X_trans_[:, :, :, :-1]), axis=3)
            del X_trans_
        return X_trans

    def _validate_X(self, X, transformer_ind):
        if not isinstance(X, np.ndarray):
            raise ValueError('Transformer {0} is wrong! `X` must be a `numpy.ndarray` object, but '
                             '`{1}` is not a `numpy.ndarray`.'.format(transformer_ind, type(X)))
        if len(X.shape) != 4:
            raise ValueError('Transformer {0} is wrong! `X` must be a 4-D array.'.format(transformer_ind))
        if X.shape[1] != 1:
            raise ValueError('Transformer {0} is wrong! '
                             'Second dimension of `X` does not equal to 1.'.format(transformer_ind))
        if X.shape[3] < 2:
            raise ValueError('Transformer {0} is wrong! Last dimension of `X` less than 2.'.format(transformer_ind))

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, None if old is None else next(transformers))
            for name, old in self.transformer_list
        ]