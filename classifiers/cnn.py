import copy
import time

import numpy
import theano
import theano.tensor as T
import lasagne

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score, roc_auc_score

from classifiers.utils import split_train_data, iterate


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layers: dict=None, dropout: float=0.5, learning_rate: float=2e-3, max_epochs_number: int=1000,
                 epochs_before_stopping: int=10, validation_fraction: float=None, beta1: float=0.9, beta2: float=0.999,
                 epsilon: float=1e-08, batch_size: int=500, batch_norm: bool=True, verbose: bool=False,
                 warm_start: bool=False, random_state=None, eval_metric: str='Logloss'):
        """ Создать свёрточную нейросеть типа LeNet, состоящую из чередующихся слоёв свёртки и макс-пулинга.

        Структура свёрточной сети задаётся словарём layers. В этом словаре должно быть три ключа "conv", "pool" и
        "dense", описывающих, соответственно, параметры свёрточных слоёв, слоёв пулинга и полносвязных слоёв
        (предполагается, что любая свёрточная сеть после серии чередующихся слоёв свёртки и подвыборки завершается хотя
        бы одним скрытым полносвязным слоем).

        Структура слоёв свёртки описывается N-элементым кортежем или списком, каждый элемент которого, в свою очередь,
        также представляет собой кортеж или список, но только из двух элементов: количества карт признаков в свёрточном
        слое и размера рецептивного поля. Размер рецептивного поля - это, как нетрудно догадаться, тоже 2-элементный
        кортеж или список, в котором заданы ширина и высота рецептивного поля для всех нейронов всех карт признаков
        свёрточного слоя. Если один из размеров рецептивного поля установлен в 0, то используется соответствующий
        размер (ширина или высота) карты признаков предыдущего слоя (или входной карты, если текущий слой - первый).
        Пример возможной структуры слоёв свёртки: (32, (5, 0), (64, (2, 0))). В данном примере указано, что свёрточная
        сеть должна иметь два слоя свёртки. Первый из слоёв включает в себя 32 карты признаков, а все нейроны этого
        слоя имеют рецептивное поле высотой 5 элементов, а шириной - во всю ширину входной карты (тем самым двумерная
        свёртка элегантным движением превращается в одномерную - только по высоте, но не по ширине). Второй из
        свёрточных слоёв включает в себя уже 64 карты признаков, а все нейроны этого слоя имеют рецептивное поле высотой
        5 элементов, а шириной во всю ширину карты признаков предшествующего слоя пулинга.

        Структура слоёв подвыборки также описывается N-элементым кортежем или списком, причём количество слоёв пулинга
        должно строго совпадать с количеством слоёв свёртки, поскольку в свёрточной сети за каждым слоем свёртки
        обязательно следует слой пулинга. Но каждый элемент, описывающий отдельный слой пулинга, отличается от элемента
        для описания слоя свёртки. Дело в том, что число карт признаков слоя пулинга всегда равно числу карт признаков
        предшествующего ему слоя свёртки, и, таким образом, дополнительно указывать это число при описании слоя пулинга
        не имеет смысла. В таком случае один элемент, описывающий отдельный слой пулинга - это 2-элеметный кортеж или
        список, задающий размер рецептивного поля всех нейронов слоя пулинга - ширину и высоту. И ширина, и высота
        должны быть положительными вещественными числами. Они определяют, во сколько раз слой пулинга будет сжимать
        карту признаков предшествующего ему слоя свёртки по высоте и ширине соответственно. Пример возможной структуры
        слоёв пулинга: ((3, 1), (2, 1)). В данном примере указано, что свёрточная сеть должна иметь два слоя пулинга.
        Первый из них сжимает карту признаков предшествующего ему слоя свёртки в три раза по высоте, а ширину оставляет
        неизменной. Второй же слой пулинга сжимает карту признаков предшествующего ему слоя свёртки в два раза по
        высоте, а ширину также оставляет неизменной.

        Структура полносвязных скрытых слоёв - это просто кортеж или список положительных целых чисел, например,
        (300, 100), что означает 300 нейронов в первом полносвязном скрытом слое и 100 нейронов во втором полносвязном
        скрытом слое. Число нейронов в выходном слое всегда определяется автоматически на основе числа распознаваемых
        классов.

        Все функции активации имеют тип ReLU, кроме функций активации нейронов выходного слоя. Для выходного же слоя
        используется либо логистическая сигмоида, если число распознаваемых классов равно двум, либо же SOFTMAX-функция
        активации, если число распознаваемых классов больше двух.

        Дропаут применяется только к полносвязной части свёрточной нейронной сети (к слоям свёртки и подвыборки
        применять дропаут бессмысленно и даже вредно).

        В качестве алгоритма обучения используется Adamax, параметры которого доступны для настройки.

        Начать обучение свёрточной нейронной сети можно как "с нуля", инициализировав все веса случайным образом, так и
        со старых значений весов, доставшихся "в наследство" от прошлых экспериментов (этим управляется параметром
        warm_start, который должен быть установлен в True, если мы хотим использовать старые значения весов в начале
        процедуры обучения).

        В процессе обучения для предотвращения переобучения (overfitting) может быть использован критерий раннего
        останова (early stopping). Этот критерий автоматически включается, если параметр validation_fraction не является
        None либо же если в метод fit() в качестве дополнительного аргумента "validation" передано контрольное
        (валидационное) множество примеров (см. комментарии к методу fit()). При включении критерия раннего останова
        процесс обучения продолжается до тех пор, пока ошибка обобщения не прекратит убывать на протяжении последних
        epochs_before_stopping эпох обучения (таким образом, параметр epochs_before_stopping определяет некое "терпение"
        нейросети в режиме раннего останова). Если же режим раннего останова не используется, то обучение нейросети
        будет продолжаться до тех пор, пока ошибка обучения (а не обобщения!) не прекратит убывать на протяжении
        того же количества epochs_before_stopping эпох подряд. Но в обоих случаях число эпох обучения не может превысить
        величину max_epochs_number (при достижении этого числа эпох обучение прекращается безотносительно выполнения
        других критериев).

        :param layers: Структура слоёв свёрточной сети (словарь с тремя ключами "conv", "pool" и "dense").
        :param dropout: Коэффициент дропаута - вещественное число больше 0, но меньше 1.
        :param learning_rate: Коэффициент скорости обучения для алгоритма Adamax - положительное вещественное число.
        :param max_epochs_number: Максимальное число эпох обучения (положительное целое число).
        :param epochs_before_stopping: Максимальное "терпение" сети в методе раннего останова (early stopping).
        :param validation_fraction: Доля примеров обучающего множества для расчёта ошибки обобщения при early stopping.
        :param beta1: Параметр алгоритма Adamax.
        :param beta2: Параметр алгоритма Adamax.
        :param epsilon: Параметр алгоритма Adamax.
        :param batch_size: Размер одного "минибатча" при обучении и эксплуатации сети - положительное целое число.
        :param batch_norm: Флажок, указывающий, надо ли использовать батч-нормализацию при обучении.
        :param verbose: Флажок, указывающий, надо ли логгировать процесс обучения (печатать на экран с помощью print).
        :param warm_start: Флажок, указывающий, надо ли начинать обучение со старых значений весов.
        :param random_state: Генератор случайных чисел (нужен, прежде всего, для отладки).
        :param eval_metric: Метрика, используемая для оценки способности к обобщению ("Logloss", "F1", "ROC-AUC").

        """
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs_number = max_epochs_number
        self.validation_fraction = validation_fraction
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.verbose = verbose
        self.epochs_before_stopping = epochs_before_stopping
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.warm_start = warm_start
        self.random_state = random_state
        self.eval_metric = eval_metric

    def fit(self, X, y, **fit_params):
        """ Обучить свёрточную нейросеть на заданном множестве примеров: входных примеров X и соответствующих меток y.

        В процессе обучения, чтобы применить критерий раннего останова, можно задать контрольное (валидационное)
        множество, на котором будет вычисляться ошибка обобщения. Это можно сделать с использованием необязательного
        аргумента validation. Если этот аргумент указан, то он должен представлять собой двухэлементый кортеж или
        список, первым элементом которого является множество входных примеров X_val, а вторым элементов - множество
        меток классов y_val. X_val, как и множество обучающих примеров X, должно быть 4-мерным numpy.ndarray-массивом,
        а y_val, как и y, должно быть 1-мерным numpy.ndarray-массивом меток классов.

        Первое измерение массивов X и X_val - это число примеров. Соответственно, первое измерение X должно быть равно
        первому (и единственному) измерению y, а первое измерение X_val - первому (и единственному) измерению y_val.

        Если необязательный аргумент validation не указан, то контрольное (валидационное) множество автоматически
        случайным образом отшипывается от обучающего множества в пропорции, заданной параметром validation_fraction.
        Если же и необязательный аргумент validation не указан, и параметр validation_fraction установлен в None, то
        критерий раннего останова не используется.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - обучающие примеры, а остальные - размеры примера).
        :param y: 1-мерный numpy.ndarray-массив меток классов для каждого примера (метка - это целое неотрицательное).
        :param validation: Необязательный параметр, задающий контрольное (валидационное) множество для early stopping.

        :return self.

        """
        self.check_params(**self.get_params(deep=False))
        X, y = self.check_train_data(X, y)
        input_structure = X.shape[1:]
        self.random_state = check_random_state(self.random_state)
        classes_list = sorted(list(set(y.tolist())))
        if self.warm_start:
            check_is_fitted(self, ['cnn_', 'predict_fn_', 'n_iter_', 'input_size_', 'loss_value_', 'classes_list_'])
            if X.shape[1:] != self.input_size_:
                raise ValueError('Samples of `X` do not correspond to the input structure! '
                                 'Got {0}, expected {1}'.format(X.shape[1:], self.input_size_))
            if self.classes_list_ != classes_list:
                raise ValueError('List of classes is wrong. Got {0}, expected {1}.'.format(
                    classes_list, self.classes_list_
                ))
            old_params = lasagne.layers.get_all_param_values(self.cnn_)
        else:
            old_params = None
        if (len(classes_list) > 2) and (self.eval_metric == 'ROC-AUC'):
            raise ValueError('You can not use `ROC-AUC` metric for early stopping '
                             'if number of classes is greater than 2.')
        if (not hasattr(self, 'cnn_train_fn_')) or (not hasattr(self, 'cnn_val_fn_')) or \
                (not hasattr(self, 'predict_fn_')) or (not hasattr(self, 'cnn_')):
            cnn_input_var = T.tensor4('inputs')
            target_var = T.ivector('targets')
            self.cnn_, _ = self.build_cnn(input_structure, len(classes_list), cnn_input_var)
            train_loss, _ = self.build_loss(len(classes_list), target_var, self.cnn_, False)
            params = lasagne.layers.get_all_params(self.cnn_, trainable=True)
            updates = lasagne.updates.adamax(train_loss, params, learning_rate=self.learning_rate, beta1=self.beta1,
                                             beta2=self.beta2, epsilon=self.epsilon)
            self.cnn_train_fn_ = theano.function([cnn_input_var, target_var], train_loss, updates=updates,
                                                 allow_input_downcast=True)
            test_loss, test_prediction = self.build_loss(len(classes_list), target_var, self.cnn_, True)
            self.cnn_val_fn_ = theano.function([cnn_input_var, target_var], test_loss, allow_input_downcast=True)
            self.predict_fn_ = theano.function([cnn_input_var], test_prediction, allow_input_downcast=True)
        if old_params is not None:
            lasagne.layers.set_all_param_values(self.cnn_, old_params)
        if 'validation' in fit_params:
            if (not isinstance(fit_params['validation'], tuple)) and (not isinstance(fit_params['validation'], list)):
                raise ValueError('Validation data are specified incorrectly!')
            if len(fit_params['validation']) != 2:
                raise ValueError('Validation data are specified incorrectly!')
            X_val, y_val = self.check_train_data(fit_params['validation'][0], fit_params['validation'][1])
            if X.shape[1:] != X_val.shape[1:]:
                raise ValueError('Validation inputs do not correspond to train inputs!')
            if set(y.tolist()) != set(y_val.tolist()):
                raise ValueError('Validation targets do not correspond to train targets!')
            train_indices = numpy.arange(0, X.shape[0], 1, numpy.int32)
            val_indices = numpy.arange(0, X_val.shape[0], 1, numpy.int32)
        elif self.validation_fraction is not None:
            n = int(round(self.validation_fraction * X.shape[0]))
            if (n <= 0) or (n >= X.shape[0]):
                raise ValueError('Train data cannot be split into train and validation subsets!')
            X_val = None
            y_val = None
            train_indices, val_indices = split_train_data(y, n, self.random_state)
        else:
            X_val = None
            y_val = None
            train_indices = numpy.arange(0, X.shape[0], 1, numpy.int32)
            val_indices = None
        if self.verbose:
            print("")
            print("Training is started...")
        best_eval_metric = None
        cur_eval_metric = None
        best_params = None
        best_epoch_ind = None
        early_stopping = False
        for epoch_ind in range(self.max_epochs_number):
            train_err = 0
            start_time = time.time()
            for batch in self.__iterate_minibatches(X, y, train_indices, shuffle=True):
                inputs, targets = batch
                train_err += self.cnn_train_fn_(inputs, targets)
            train_err /= train_indices.shape[0]
            val_err = 0.0
            if val_indices is None:
                if best_eval_metric is None:
                    best_epoch_ind = epoch_ind
                    best_eval_metric = train_err
                    best_params = lasagne.layers.get_all_param_values(self.cnn_)
                elif train_err < best_eval_metric:
                    best_epoch_ind = epoch_ind
                    best_eval_metric = train_err
                    best_params = lasagne.layers.get_all_param_values(self.cnn_)
            else:
                val_err = 0
                if X_val is None:
                    for batch in self.__iterate_minibatches(X, y, val_indices, shuffle=False):
                        inputs, targets = batch
                        val_err += self.cnn_val_fn_(inputs, targets)
                else:
                    for batch in self.__iterate_minibatches(X_val, y_val, val_indices, shuffle=False):
                        inputs, targets = batch
                        val_err += self.cnn_val_fn_(inputs, targets)
                val_err /= val_indices.shape[0]
                if self.eval_metric == 'Logloss':
                    cur_eval_metric = val_err
                    if best_eval_metric is None:
                        best_epoch_ind = epoch_ind
                        best_eval_metric = cur_eval_metric
                        best_params = lasagne.layers.get_all_param_values(self.cnn_)
                    elif cur_eval_metric < best_eval_metric:
                        best_epoch_ind = epoch_ind
                        best_eval_metric = cur_eval_metric
                        best_params = lasagne.layers.get_all_param_values(self.cnn_)
                else:
                    if self.eval_metric == 'F1':
                        if X_val is None:
                            cur_eval_metric = f1_score(
                                y[val_indices],
                                self.__predict(X[val_indices], len(classes_list)),
                                average=('binary' if len(classes_list) < 3 else 'macro')
                            )
                        else:
                            cur_eval_metric = f1_score(
                                y_val,
                                self.__predict(X_val, len(classes_list)),
                                average=('binary' if len(classes_list) < 3 else 'macro')
                            )
                    else:
                        if X_val is None:
                            cur_eval_metric = roc_auc_score(
                                y[val_indices],
                                self.__predict_proba(X[val_indices], len(classes_list))[:, 1]
                            )
                        else:
                            cur_eval_metric = roc_auc_score(
                                y_val,
                                self.__predict_proba(X_val, len(classes_list))[:, 1]
                            )
                    if best_eval_metric is None:
                        best_epoch_ind = epoch_ind
                        best_eval_metric = cur_eval_metric
                        best_params = lasagne.layers.get_all_param_values(self.cnn_)
                    elif cur_eval_metric > best_eval_metric:
                        best_epoch_ind = epoch_ind
                        best_eval_metric = cur_eval_metric
                        best_params = lasagne.layers.get_all_param_values(self.cnn_)
            if self.verbose:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch_ind + 1, self.max_epochs_number, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err))
                if val_indices is not None:
                    print("  validation loss:\t\t{:.6f}".format(val_err))
                    if self.eval_metric != 'Logloss':
                        print("  validation {}:\t\t{:.6f}".format(self.eval_metric, cur_eval_metric))
            if best_epoch_ind is not None:
                if (epoch_ind - best_epoch_ind) >= self.epochs_before_stopping:
                    early_stopping = True
                    break
        if best_params is None:
            raise ValueError('The multilayer perceptron cannot be trained!')
        self.loss_value_ = best_eval_metric
        if self.warm_start:
            self.n_iter_ += (best_epoch_ind + 1)
        else:
            self.n_iter_ = best_epoch_ind + 1
        lasagne.layers.set_all_param_values(self.cnn_, best_params)
        del best_params
        self.input_size_ = input_structure
        if self.verbose:
            if early_stopping:
                print('Training is stopped according to the early stopping criterion.')
            else:
                print('Training is stopped according to the exceeding of maximal epochs number.')
        self.classes_list_ = classes_list
        return self

    def __predict(self, X, n_classes):
        """ Распознать обученной нейросетью заданное множество входных примеров X.

        Перед распознаванием ничего не проверять - ни корректность параметров нейросети, ни правильность входных данных.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - тестовые примеры, а остальные - размеры примера).
        :param n_classes: количество распознаваемых классов.

        :return 1-мерный numpy.ndarray-массив распознанных меток классов(целых чисел), равный по длине 1-му измерению X.

        """
        n_samples = X.shape[0]
        y_pred = numpy.zeros((n_samples,), dtype=numpy.int32)
        if n_classes > 2:
            sample_ind = 0
            for batch in self.__iterate_minibatches_for_prediction(X):
                inputs = batch
                outputs = self.predict_fn_(inputs)
                n_outputs = outputs.shape[0]
                if sample_ind + n_outputs <= n_samples:
                    y_pred[sample_ind:(sample_ind + n_outputs)] = outputs.argmax(axis=1).astype(y_pred.dtype)
                else:
                    y_pred[sample_ind:n_samples] = outputs[:(n_samples - sample_ind)].argmax(axis=1).astype(
                        y_pred.dtype)
                sample_ind += n_outputs
        else:
            sample_ind = 0
            for batch in self.__iterate_minibatches_for_prediction(X):
                inputs = batch
                outputs = self.predict_fn_(inputs)
                n_outputs = outputs.shape[0]
                if sample_ind + n_outputs <= n_samples:
                    y_pred[sample_ind:(sample_ind + n_outputs)] = (outputs >= 0.5).astype(y_pred.dtype)
                else:
                    y_pred[sample_ind:n_samples] = (outputs[:(n_samples - sample_ind)] >= 0.5).astype(y_pred.dtype)
                sample_ind += n_outputs
        return y_pred

    def predict(self, X):
        """ Распознать обученной нейросетью заданное множество входных примеров X.

        Перед распознаванием проверить корректность установки всех параметров нейросети и правильность задания множества
        входных примеров.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - тестовые примеры, а остальные - размеры примера).

        :return 1-мерный numpy.ndarray-массив распознанных меток классов(целых чисел), равный по длине 1-му измерению X.

        """
        check_is_fitted(self, ['cnn_', 'predict_fn_', 'n_iter_', 'input_size_', 'loss_value_', 'classes_list_'])
        X = self.check_input_data(X)
        return self.__predict(X, len(self.classes_list_))

    def __predict_proba(self, X, n_classes):
        """ Вычислить вероятности распознавания классов для заданного множества входных примеров X.

        Перед вычислением вероятностей ничего не проверять - ни корректность параметров нейросети, ни правильность
        входных данных.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - тестовые примеры, а остальные - размеры примера).
        :param n_classes: количество распознаваемых классов.

        :return 2-мерный numpy.ndarray-массив, число строк которого равно 1-му измерению X, а столбцов - числу классов.

        """
        n_samples = X.shape[0]
        sample_ind = 0
        if n_classes > 2:
            probabilities = numpy.empty((n_samples, n_classes), dtype=numpy.float32)
            for batch in self.__iterate_minibatches_for_prediction(X):
                inputs = batch
                outputs = self.predict_fn_(inputs)
                n_outputs = outputs.shape[0]
                if sample_ind + n_outputs <= n_samples:
                    probabilities[sample_ind:(sample_ind + n_outputs)] = outputs
                else:
                    probabilities[sample_ind:n_samples] = outputs[:(n_samples - sample_ind)]
                sample_ind += n_outputs
            res = probabilities
        else:
            probabilities = numpy.empty((n_samples,), dtype=numpy.float32)
            for batch in self.__iterate_minibatches_for_prediction(X):
                inputs = batch
                outputs = self.predict_fn_(inputs)
                n_outputs = outputs.shape[0]
                if sample_ind + n_outputs <= n_samples:
                    probabilities[sample_ind:(sample_ind + n_outputs)] = outputs
                else:
                    probabilities[sample_ind:n_samples] = outputs[:(n_samples - sample_ind)]
                sample_ind += n_outputs
            probabilities = probabilities.reshape((probabilities.shape[0], 1))
            res = numpy.hstack((1.0 - probabilities, probabilities))
        return res

    def predict_proba(self, X):
        """ Вычислить вероятности распознавания классов для заданного множества входных примеров X.

        Перед вычислением вероятностей проверить корректность установки всех параметров нейросети и правильность задания
        множества входных примеров.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - тестовые примеры, а остальные - размеры примера).

        :return 2-мерный numpy.ndarray-массив, число строк которого равно 1-му измерению X, а столбцов - числу классов.

        """
        check_is_fitted(self, ['cnn_', 'predict_fn_', 'n_iter_', 'input_size_', 'loss_value_', 'classes_list_'])
        X = self.check_input_data(X)
        return self.__predict_proba(X, len(self.classes_list_))

    def predict_log_proba(self, X):
        """ Вычислить логарифмы вероятностей распознавания классов для заданного множества входных примеров X.

        :param X: 4-мерный numpy.ndarray-массив (1-е измерение - тестовые примеры, а остальные - размеры примера).

        :return 2-мерный numpy.ndarray-массив, число строк которого равно 1-му измерению X, а столбцов - числу классов.

        """
        return numpy.log(self.predict_proba(X))

    def get_params(self, deep=True):
        """ Получить словарь управляющих параметров нейросети.

        Данный метод используется внутри sklearn.pipeline.Pipeline, sklearn.model_selection.GridSearchCV и пр.
        Соответствено, если мы хотим насладиться всей мощью scikit-learn и использовать наш класс там, то данный метод
        нужно корректно реализовать.

        :return словарь управляющих параметров нейросети (без параметров, настроенных по итогам обучения).

        """
        return {'layers': copy.deepcopy(self.layers) if deep else self.layers, 'dropout': self.dropout,
                'learning_rate': self.learning_rate, 'max_epochs_number': self.max_epochs_number,
                'validation_fraction': self.validation_fraction, 'epochs_before_stopping': self.epochs_before_stopping,
                'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'batch_size': self.batch_size,
                'verbose': self.verbose, 'batch_norm': self.batch_norm, 'warm_start': self.warm_start,
                'eval_metric': self.eval_metric}

    def set_params(self, **parameters):
        """ Установить новые значения управляющих параметров нейросети из словаря.

        Данный метод используется внутри sklearn.pipeline.Pipeline, sklearn.model_selection.GridSearchCV и пр.
        Соответствено, если мы хотим насладиться всей мощью scikit-learn и использовать наш класс там, то данный метод
        нужно корректно реализовать.

        :param parameters: Названия и значения устанавливаемых параметров, заданные словарём.

        :return self

        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

    @staticmethod
    def check_params(**kwargs):
        """ Проверить корректность значений всех возможных параметров и, если что, бросить ValueError. """
        if not 'layers' in kwargs:
            raise ValueError('Structure of hidden layers is not specified!')
        if not isinstance(kwargs['layers'], dict):
            raise ValueError('Structure of hidden layers must be dictionary consisting from three items!')
        if len(kwargs['layers']) != 3:
            raise ValueError('Structure of hidden layers must be dictionary consisting from three items!')
        if 'conv' not in kwargs['layers']:
            raise ValueError('Description of convolution layers (`conv` key) cannot be found in the `layers` dict!')
        conv_layers = kwargs['layers']['conv']
        if 'pool' not in kwargs['layers']:
            raise ValueError('Description of pooling layers (`pool` key) cannot be found in the `layers` dict!')
        pooling_layers = kwargs['layers']['pool']
        if 'dense' not in kwargs['layers']:
            raise ValueError('Description of dense layers (`dense` key) cannot be found in the `layers` dict!')
        dense_layers = kwargs['layers']['dense']
        if (not isinstance(conv_layers, tuple)) and (not isinstance(conv_layers, list)):
            raise ValueError('Structure of convolution layers must be list or tuple!')
        if len(conv_layers) < 1:
            raise ValueError('List of convolution layers is empty!')
        if (not isinstance(pooling_layers, tuple)) and (not isinstance(pooling_layers, list)):
            raise ValueError('Structure of pooling layers must be list or tuple!')
        if len(pooling_layers) < 1:
            raise ValueError('List of pooling layers is empty!')
        if (not isinstance(dense_layers, tuple)) and (not isinstance(dense_layers, list)):
            raise ValueError('Structure of dense layers must be list or tuple!')
        if len(dense_layers) < 1:
            raise ValueError('List of dense layers is empty!')
        if len(conv_layers) != len(pooling_layers):
            raise ValueError('Number of convolution layers must be equal to number of pooling layers!')
        for ind in range(len(conv_layers)):
            err_msg = 'Structure of convolution layer {0} is wrong!'.format(ind + 1)
            if (not isinstance(conv_layers[ind], tuple)) and (not isinstance(conv_layers[ind], list)):
                raise ValueError(err_msg)
            if len(conv_layers[ind]) != 2:
                raise ValueError(err_msg)
            if not isinstance(conv_layers[ind][0], int):
                raise ValueError(err_msg)
            if conv_layers[ind][0] < 1:
                raise ValueError(err_msg)
            if (not isinstance(conv_layers[ind][1], tuple)) and (not isinstance(conv_layers[ind][1], list)):
                raise ValueError(err_msg)
            if len(conv_layers[ind][1]) != 2:
                raise ValueError(err_msg)
            if not isinstance(conv_layers[ind][1][0], int):
                raise ValueError(err_msg)
            if conv_layers[ind][1][0] < 0:
                raise ValueError(err_msg)
            if not isinstance(conv_layers[ind][1][1], int):
                raise ValueError(err_msg)
            if conv_layers[ind][1][1] < 0:
                raise ValueError(err_msg)
            if (conv_layers[ind][1][0] <= 0) and (conv_layers[ind][1][1] <= 0):
                raise ValueError(err_msg)
            err_msg = 'Structure of pooling layer {0} is wrong!'.format(ind + 1)
            if (not isinstance(pooling_layers[ind], tuple)) and (not isinstance(pooling_layers[ind], list)):
                raise ValueError(err_msg)
            if len(pooling_layers[ind]) != 2:
                raise ValueError(err_msg)
            if pooling_layers[ind][0] < 0:
                raise ValueError(err_msg)
            if pooling_layers[ind][1] < 0:
                raise ValueError(err_msg)
            if (pooling_layers[ind][0] <= 0) and (pooling_layers[ind][1] <= 0):
                raise ValueError(err_msg)
            receptive_field_for_pool_layer = (
                pooling_layers[ind][0] if pooling_layers[ind][0] > 0 else 1,
                pooling_layers[ind][1] if pooling_layers[ind][1] > 0 else 1
            )
            if (receptive_field_for_pool_layer[0] < 1) or (receptive_field_for_pool_layer[1] < 1):
                raise ValueError(err_msg)
        n_dense_layers = len(dense_layers)
        for layer_ind in range(n_dense_layers):
            if (not isinstance(dense_layers[layer_ind], int)) or (dense_layers[layer_ind] < 1):
                raise ValueError('Size of fully-connected layer {0} is inadmissible!'.format(layer_ind))
        if 'dropout' not in kwargs:
            raise ValueError('Dropout probability is not specified!')
        if (kwargs['dropout'] <= 0.0) or (kwargs['dropout'] >= 1.0):
            raise ValueError('Dropout probability is wrong!')
        if 'learning_rate' not in kwargs:
            raise ValueError('Learning rate is not specified!')
        if kwargs['learning_rate'] <= 0.0:
            raise ValueError('Learning rate must be positive value!')
        if 'max_epochs_number' not in kwargs:
            raise ValueError('Maximal number of train epochs is not specified!')
        if (not isinstance(kwargs['max_epochs_number'], int)) or (kwargs['max_epochs_number'] <= 0):
            raise ValueError('Maximal number of train epochs must be positive integer value!')
        if 'validation_fraction' not in kwargs:
            raise ValueError('Validation fraction is not specified!')
        if kwargs['validation_fraction'] is not None:
            if (kwargs['validation_fraction'] <= 0.0) or (kwargs['validation_fraction'] >= 1.0):
                raise ValueError('Validation fraction must be in (0.0, 1.0) or None!')
        if not 'beta1' in kwargs:
            raise ValueError('Beta1 for the Adamax algorithm is not specified!')
        if (kwargs['beta1'] < 0.0) or (kwargs['beta1'] >= 1.0):
            raise ValueError('Beta1 for the Adamax algorithm must be in [0.0, 1.0)!')
        if not 'beta2' in kwargs:
            raise ValueError('Beta2 for the Adamax algorithm is not specified!')
        if (kwargs['beta2'] < 0.0) or (kwargs['beta2'] >= 1.0):
            raise ValueError('Beta2 for the Adamax algorithm must be in [0.0, 1.0)!')
        if 'epsilon' not in kwargs:
            raise ValueError('Epsilon for the Adamax algorithm is not specified!')
        if kwargs['epsilon'] <= 0.0:
            raise ValueError('Epsilon for the Adamax algorithm must be positive value!')
        if not 'batch_size' in kwargs:
            raise ValueError('Batch size is not specified!')
        if (not isinstance(kwargs['batch_size'], int)) or (kwargs['batch_size'] <= 0):
            raise ValueError('Batch size must be positive integer value!')
        if not 'epochs_before_stopping' in kwargs:
            raise ValueError('Maximal number of consecutive epochs when validation score is not improving '
                             'is not specified!')
        if (not isinstance(kwargs['epochs_before_stopping'], int)) or (kwargs['epochs_before_stopping'] <= 0):
            raise ValueError('Maximal number of consecutive epochs when validation score is not improving must be '
                             'positive integer value!')
        if kwargs['epochs_before_stopping'] > kwargs['max_epochs_number']:
            raise ValueError('Maximal number of consecutive epochs when validation score is not improving must be '
                             'positive integer value!')
        if not 'batch_norm' in kwargs:
            raise ValueError('Flag of the batch normalization is not specified!')
        if not isinstance(kwargs['batch_norm'], bool):
            raise ValueError('Flag of the batch normalization must be boolean value!')
        if not 'warm_start' in kwargs:
            raise ValueError('Flag of the warm start is not specified!')
        if not isinstance(kwargs['warm_start'], bool):
            raise ValueError('Flag of the warm start must be boolean value!')
        if not 'verbose' in kwargs:
            raise ValueError('Flag of the verbose mode is not specified!')
        if (not isinstance(kwargs['verbose'], bool)) and (not isinstance(kwargs['verbose'], int)):
            raise ValueError('Flag of the verbose mode must be boolean or integer value!')
        if not 'eval_metric' in kwargs:
            raise ValueError('Metric for evaluation and early stopping is not specified!')
        if not isinstance(kwargs['eval_metric'], str):
            raise ValueError('Metric for evaluation and early stopping must be a string value!')
        if kwargs['eval_metric'] not in {'Logloss', 'F1', 'ROC-AUC'}:
            raise ValueError('"{0}" is unknown metric for evaluation and early stopping! '
                             'We expect "Logloss", "F1" or "ROC-AUC".'.format(kwargs['eval_metric']))

    def check_train_data(self, X, y):
        """ Проверить корректность обучающего (или тестового) множества входных примеров X и меток классов y.

        Если что-то пошло не так, то бросить ValueError.

        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=[numpy.float32, numpy.float64, numpy.uint8], allow_nd=True)
        if len(X.shape) != 4:
            raise ValueError('`X` must be a 4-D array (samples, input maps, rows of input map, columns of input map)!')
        for sample_ind in range(y.shape[0]):
            if y[sample_ind] < 0:
                raise ValueError('Target values must be non-negative integer numbers!')
        if set(y.tolist()) != set(range(int(y.max()) + 1)):
            raise ValueError('Target values must be non-negative integer numbers!')
        return X, y

    def check_input_data(self, X):
        """ Проверить корректность множества входных примеров X и бросить ValueError в случае ошибки. """
        X = check_array(X, accept_sparse=False, dtype=[numpy.float32, numpy.float64, numpy.uint8], allow_nd=True)
        if len(X.shape) != 4:
            raise ValueError('`X` must be 4D array (samples, input maps, rows of input map, columns of input map)!')
        if X.shape[1:] != self.input_size_:
            raise ValueError('Samples of `X` do not correspond to the input structure! '
                             'Got {0}, expected {1}'.format(X.shape[1:], self.input_size_))
        return X

    def build_cnn(self, input_structure, number_of_classes, input_var, trainable_shared_params=None):
        """ Построить вычислительный граф нейросети средствами Theano/Lasagne.

        :param input_structure - 3-элементный кортеж, задающий количество входных карт, их высоту и ширину.
        :param number_of_classes - число распознаваемых классов.
        :param input_var - символьная входная переменная для вычислительного графа Theano.
        :param trainable_shared_params - расшариваемые параметры нейросети-близнеца (или None, если близнецов нет).

        :return кортеж из двух вычислительных графов Theano: для всей сети в целом и для сети без выходного слоя.

        """
        l_in = lasagne.layers.InputLayer(
            shape=(None, input_structure[0], input_structure[1], input_structure[2]),
            input_var=input_var
        )
        input_size = (input_structure[1], input_structure[2])
        conv_layers = self.layers['conv']
        pooling_layers = self.layers['pool']
        dense_layers = self.layers['dense']
        if conv_layers[0][0] <= 0:
            raise ValueError('Convolution layer 1: {0} is wrong number of feature maps!'.format(conv_layers[0][0]))
        receptive_field_for_conv_layer = (
            conv_layers[0][1][0] if conv_layers[0][1][0] > 0 else input_size[0],
            conv_layers[0][1][1] if conv_layers[0][1][1] > 0 else input_size[1]
        )
        if (receptive_field_for_conv_layer[0] <= 0) or (receptive_field_for_conv_layer[1] <= 0):
            raise ValueError('Convolution layer 1: ({0}, {1}) is wrong size of receptive field!'.format(
                receptive_field_for_conv_layer[0], receptive_field_for_conv_layer[1]
            ))
        feature_map_for_conv_layer = (
            input_size[0] + 1 - receptive_field_for_conv_layer[0],
            input_size[1] + 1 - receptive_field_for_conv_layer[1]
        )
        if (feature_map_for_conv_layer[0] <= 0) or (feature_map_for_conv_layer[1] <= 0):
            raise ValueError('Convolution layer 1: ({0}, {1}) is wrong size of feature map!'.format(
                feature_map_for_conv_layer[0], feature_map_for_conv_layer[1]
            ))
        if self.batch_norm:
            l_conv = lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    l_in, num_filters=conv_layers[0][0],
                    filter_size=receptive_field_for_conv_layer,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                    else trainable_shared_params[0],
                    name='l_conv_1'
                )
            )
        else:
            l_conv = lasagne.layers.Conv2DLayer(
                l_in, num_filters=conv_layers[0][0],
                filter_size=receptive_field_for_conv_layer,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                else trainable_shared_params[0],
                b=lasagne.init.Constant(0.0) if trainable_shared_params is None else trainable_shared_params[1],
                name='l_conv_1'
            )
        receptive_field_for_pool_layer = (
            pooling_layers[0][0] if pooling_layers[0][0] > 0 else 1,
            pooling_layers[0][1] if pooling_layers[0][1] > 0 else 1
        )
        if (receptive_field_for_pool_layer[0] <= 0) or (receptive_field_for_pool_layer[1] <= 0):
            raise ValueError('Pooling layer 1: ({0}, {1}) is wrong size of receptive field!'.format(
                receptive_field_for_pool_layer[0], receptive_field_for_pool_layer[1]
            ))
        feature_map_for_pool_layer = (
            feature_map_for_conv_layer[0] // receptive_field_for_pool_layer[0],
            feature_map_for_conv_layer[1] // receptive_field_for_pool_layer[1]
        )
        if (feature_map_for_pool_layer[0] <= 0) or (feature_map_for_pool_layer[1] <= 0):
            raise ValueError('Pooling layer 1: ({0}, {1}) is wrong size of feature map!'.format(
                feature_map_for_pool_layer[0], feature_map_for_pool_layer[1]
            ))
        l_pool = lasagne.layers.Pool2DLayer(
            l_conv,
            pool_size=receptive_field_for_pool_layer,
            name='l_pool_1'
        )
        input_size = feature_map_for_pool_layer
        for ind in range(len(conv_layers) - 1):
            if conv_layers[ind + 1][0] <= 0:
                raise ValueError('Convolution layer {0}: {1} is wrong number of feature maps!'.format(
                    ind + 2, conv_layers[ind + 1][0]
                ))
            receptive_field_for_conv_layer = (
                conv_layers[ind + 1][1][0] if conv_layers[ind + 1][1][0] > 0 else input_size[0],
                conv_layers[ind + 1][1][1] if conv_layers[ind + 1][1][1] > 0 else input_size[1]
            )
            if (receptive_field_for_conv_layer[0] <= 0) or (receptive_field_for_conv_layer[1] <= 0):
                raise ValueError('Convolution layer {0}: ({1}, {2}) is wrong size of receptive field!'.format(
                    ind + 2, receptive_field_for_conv_layer[0], receptive_field_for_conv_layer[1]
                ))
            feature_map_for_conv_layer = (
                input_size[0] + 1 - receptive_field_for_conv_layer[0],
                input_size[1] + 1 - receptive_field_for_conv_layer[1]
            )
            if (feature_map_for_conv_layer[0] <= 0) or (feature_map_for_conv_layer[1] <= 0):
                raise ValueError('Convolution layer {0}: ({1}, {2}) is wrong size of feature map!'.format(
                    ind + 2, feature_map_for_conv_layer[0], feature_map_for_conv_layer[1]
                ))
            if self.batch_norm:
                l_conv = lasagne.layers.batch_norm(
                    lasagne.layers.Conv2DLayer(
                        l_pool, num_filters=conv_layers[ind + 1][0],
                        filter_size=receptive_field_for_conv_layer,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                        else trainable_shared_params[(ind + 1) * 3],
                        name='l_conv_{0}'.format(ind + 2)
                    )
                )
            else:
                l_conv = lasagne.layers.Conv2DLayer(
                    l_pool, num_filters=conv_layers[ind + 1][0],
                    filter_size=receptive_field_for_conv_layer,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                    else trainable_shared_params[(ind + 1) * 2],
                    b=lasagne.init.Constant(0.0) if trainable_shared_params is None
                    else trainable_shared_params[(ind + 1) * 2 + 1],
                    name='l_conv_{0}'.format(ind + 2)
                )
            receptive_field_for_pool_layer = (
                pooling_layers[ind + 1][0] if pooling_layers[ind + 1][0] > 0 else 1,
                pooling_layers[ind + 1][1] if pooling_layers[ind + 1][1] > 0 else 1
            )
            if (feature_map_for_pool_layer[0] <= 0) or (feature_map_for_pool_layer[1] <= 0):
                raise ValueError('Pooling layer {0}: ({1}, {2}) is wrong size of feature map!'.format(
                    ind + 2, feature_map_for_pool_layer[0], feature_map_for_pool_layer[1]
                ))
            feature_map_for_pool_layer = (
                feature_map_for_conv_layer[0] // receptive_field_for_pool_layer[0],
                feature_map_for_conv_layer[1] // receptive_field_for_pool_layer[1]
            )
            if (feature_map_for_pool_layer[0] <= 0) or (feature_map_for_pool_layer[1] <= 0):
                raise ValueError('Pooling layer {0}: ({1}, {2}) is wrong size of feature map!'.format(
                    ind + 2, feature_map_for_pool_layer[0], feature_map_for_pool_layer[1]
                ))
            l_pool = lasagne.layers.Pool2DLayer(
                l_conv,
                pool_size=receptive_field_for_pool_layer,
                name='l_pool_{0}'.format(ind + 2)
            )
            input_size = feature_map_for_pool_layer
        layer_ind = len(conv_layers) * 3 if self.batch_norm else len(conv_layers) * 2
        l_in_drop = lasagne.layers.DropoutLayer(l_pool, p=self.dropout)
        if self.batch_norm:
            l_hid_old = lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(
                    l_in_drop, num_units=dense_layers[0],
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                    else trainable_shared_params[layer_ind],
                    name='l_dense_1'
                )
            )
            layer_ind += 3
        else:
            l_hid_old = lasagne.layers.DenseLayer(
                l_in_drop, num_units=dense_layers[0],
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                else trainable_shared_params[layer_ind],
                b=lasagne.init.Constant(0.0) if trainable_shared_params is None
                else trainable_shared_params[layer_ind + 1],
                name='l_dense_1'
            )
            layer_ind += 2
        last_real_layer = l_hid_old
        l_hid_old_drop = lasagne.layers.DropoutLayer(l_hid_old, p=self.dropout)
        for ind in range(len(dense_layers) - 1):
            if self.batch_norm:
                l_hid_new = lasagne.layers.batch_norm(
                    lasagne.layers.DenseLayer(
                        l_hid_old_drop, num_units=dense_layers[ind + 1],
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                        else trainable_shared_params[layer_ind],
                        name='l_dense_{0}'.format(ind + 2)
                    )
                )
                layer_ind += 3
            else:
                l_hid_new = lasagne.layers.DenseLayer(
                    l_hid_old_drop, num_units=dense_layers[ind + 1],
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(gain='relu') if trainable_shared_params is None
                    else trainable_shared_params[layer_ind],
                    b=lasagne.init.Constant(0.0) if trainable_shared_params is None
                    else trainable_shared_params[layer_ind + 1],
                    name='l_dense_{0}'.format(ind + 2)
                )
                layer_ind += 2
            last_real_layer = l_hid_new
            l_hid_new_drop = lasagne.layers.DropoutLayer(l_hid_new, p=self.dropout)
            l_hid_old_drop = l_hid_new_drop
        last_layer = l_hid_old_drop
        if number_of_classes > 2:
            cnn_output = lasagne.layers.DenseLayer(
                last_layer, num_units=number_of_classes,
                nonlinearity=lasagne.nonlinearities.softmax,
                W=lasagne.init.GlorotUniform() if trainable_shared_params is None else trainable_shared_params[-2],
                b=lasagne.init.Constant(0.0) if trainable_shared_params is None else trainable_shared_params[-1],
                name='l_cnn'
            )
        else:
            cnn_output = lasagne.layers.DenseLayer(
                last_layer, num_units=1,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform() if trainable_shared_params is None else trainable_shared_params[-2],
                b=lasagne.init.Constant(0.0) if trainable_shared_params is None else trainable_shared_params[-1],
                name='l_cnn'
            )
        return cnn_output, last_real_layer

    def build_loss(self, number_of_classes, target_var, cnn, deterministic):
        """ Построить вычислительный граф для функции потерь и классификационной функции средствами Theano/Lasagne.

        :param number_of_classes: Число распознаваемых классов.
        :param target_var: Символьная переменная Theano, задающая желаемые метки классов для обучения/тестирования.
        :param cnn: вычислительный граф Theano для всей нейросети.
        :param deterministic: булевый флаг, определяющий режим работы (True - тестирование, False - обучение).

        :return 2-элементный кортеж: построенные графы для функции потерь и для классификационной функции.

        """
        if number_of_classes > 2:
            prediction = lasagne.layers.get_output(cnn)
            loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
            output_prediction = lasagne.layers.get_output(cnn, deterministic=deterministic)
        else:
            prediction = lasagne.layers.get_output(cnn)
            loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
            output_prediction = T.flatten(lasagne.layers.get_output(cnn, deterministic=deterministic))
        loss = loss.sum()
        return loss, output_prediction

    def dump_all(self):
        """ Выполнить сериализацию нейросети в словарь (dict).

        Метод выгружает значения всех параметров нейросети в словарь, ключами которого являются названия параметров,
        а значениями - соответственно, значения. В сериализации участвуют абсолютно все параметры, кроме random_state,
        т.е. и управляющие параметры, задаваемые, например, в конструкторе, и настраиваемые параметры, значения которых
        устанавливаются по итогам обучения (веса нейросети, распознаваемые классы и прочее).

        При сериализации выполняется копирование (а не передача по ссылка) всех составных структур данных.

        :return: словарь для всех параметров нейросети.

        """
        try:
            check_is_fitted(self, ['cnn_', 'predict_fn_', 'n_iter_', 'input_size_', 'loss_value_', 'classes_list_'])
            is_trained = True
        except:
            is_trained = False
        params = self.get_params(True)
        if is_trained:
            params['weights_and_biases'] = lasagne.layers.get_all_param_values(self.cnn_)
            params['loss_value_'] = self.loss_value_
            params['n_iter_'] = self.n_iter_
            params['input_size_'] = self.input_size_
            params['classes_list_'] = copy.copy(self.classes_list_)
        return params

    def load_all(self, new_params):
        """ Выполнить десериализацию нейросети из словаря (dict).

        Метод проверяет корректность всех параметров нейросети, заданных во входном словаре, и в случае успешной
        проверки переносит эти значения в нейросеть (в случае неудачи бросает исключение ValueError). В десериализации
        участвуют абсолютно все параметры, кроме random_state, т.е. и управляющие параметры, задаваемые, например, в
        конструкторе, и настраиваемые параметры, значения которых устанавливаются по итогам обучения (веса нейросети,
        распознаваемые классы и прочее).

        При десериализации выполняется копирование (а не передача по ссылка) всех составных структур данных.

        :param new_params: словарь (dict) со всеми параметрами нейросети для десериализации.

        :return: self

        """
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected {0}.'.format(type({0: 1})))
        self.check_params(**new_params)
        expected_param_keys = {'layers', 'dropout', 'learning_rate', 'max_epochs_number', 'validation_fraction',
                               'epochs_before_stopping', 'beta1', 'beta2', 'epsilon', 'batch_size', 'verbose',
                               'batch_norm', 'warm_start', 'eval_metric'}
        params_after_training = {'weights_and_biases', 'loss_value_', 'n_iter_', 'input_size_', 'classes_list_'}
        is_fitted = len(set(new_params.keys())) > len(expected_param_keys)
        if is_fitted:
            if set(new_params.keys()) != (expected_param_keys | params_after_training):
                raise ValueError('`new_params` does not contain the expected keys!')
        self.layers = copy.deepcopy(new_params['layers'])
        self.dropout = new_params['dropout']
        self.learning_rate = new_params['learning_rate']
        self.max_epochs_number = new_params['max_epochs_number']
        self.validation_fraction = new_params['validation_fraction']
        self.beta1 = new_params['beta1']
        self.beta2 = new_params['beta2']
        self.epsilon = new_params['epsilon']
        self.verbose = new_params['verbose']
        self.epochs_before_stopping = new_params['epochs_before_stopping']
        self.batch_size = new_params['batch_size']
        self.batch_norm = new_params['batch_norm']
        self.warm_start = new_params['warm_start']
        self.eval_metric = new_params['eval_metric']
        if getattr(self, 'random_state', None) is None:
            self.random_state = None
        if is_fitted:
            if not isinstance(new_params['loss_value_'], float):
                raise ValueError('`new_params` is wrong! Generalization loss `loss_value_` must be '
                                 'floating-point number!')
            if not isinstance(new_params['n_iter_'], int):
                raise ValueError('`new_params` is wrong! Generalization loss `n_iter_` must be positive integer!')
            if new_params['n_iter_'] <= 0:
                raise ValueError('`new_params` is wrong! Generalization loss `n_iter_` must be positive integer!')
            if (not isinstance(new_params['input_size_'], tuple)) and (not isinstance(new_params['input_size_'], list)):
                raise ValueError('`new_params` is wrong! All input data sizes `input_size_` must be list or tuple!')
            if len(new_params['input_size_']) != 3:
                raise ValueError('`new_params` is wrong! All input data sizes `input_size_` must be 3-D sequence!')
            for cur in new_params['input_size_']:
                if not isinstance(cur, int):
                    raise ValueError('`new_params` is wrong! Each input data size `input_size_` must be '
                                     'positive integer number!')
                if cur <= 0:
                    raise ValueError('`new_params` is wrong! Each input data size `input_size_` must be '
                                     'positive integer number!')
            if (not isinstance(new_params['classes_list_'], list)) and \
                    (not isinstance(new_params['classes_list_'], tuple)):
                raise ValueError('`new_params` is wrong! The classes list `classes_list_` must be list or tuple!')
            if len(new_params['classes_list_']) < 2:
                raise ValueError('`new_params` is wrong! The classes list `classes_list_` must consist from '
                                 'two or more classes!')
            self.random_state = check_random_state(self.random_state)
            self.loss_value_ = new_params['loss_value_']
            self.n_iter_ = new_params['n_iter_']
            self.classes_list_ = copy.copy(new_params['classes_list_'])
            self.input_size_ = copy.copy(new_params['input_size_'])
            cnn_input_var = T.tensor4('inputs')
            target_var = T.ivector('targets')
            self.cnn_, _ = self.build_cnn(self.input_size_, len(self.classes_list_), cnn_input_var)
            _, test_prediction = self.build_loss(len(self.classes_list_), target_var, self.cnn_, True)
            lasagne.layers.set_all_param_values(self.cnn_, new_params['weights_and_biases'])
            self.predict_fn_ = theano.function([cnn_input_var], test_prediction, allow_input_downcast=True)
        return self

    def __iterate_minibatches(self, inputs, targets, indices, shuffle=False):
        """ Итерироваться "минибатчами" по датасету - входным примерам inputs и соответствующим меткам классов targets.

        :param inputs: Входные примеры X.
        :param targets: Метки классов y.
        :param indices: Индексы интересных нам примеров, участвующих в итерировании.
        :param shuffle: Булевый флажок, указывающий, итерироваться случайно или всё же последовательно.

        :return Итератор (каждый элемент: "минибатч" из batch_size входных примеров и соответствующих им меток классов).

        """
        for indices_in_batch in iterate(indices, self.batch_size, shuffle, self.random_state if shuffle else None):
            yield inputs[indices_in_batch], targets[indices_in_batch]

    def __iterate_minibatches_for_prediction(self, inputs):
        """ Итерироваться "минибатчами" по входным примерам inputs.

        :param inputs: Входные примеры X.

        :return Итератор (каждый элемент: "минибатч" из batch_size входных примеров).

        """
        for indices_in_batch in iterate(numpy.arange(0, inputs.shape[0], 1, numpy.int32), self.batch_size, False, None):
            yield inputs[indices_in_batch]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.load_all(self.dump_all())
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.load_all(self.dump_all())
        return result

    def __getstate__(self):
        """ Нужно для сериализации через pickle. """
        return self.dump_all()

    def __setstate__(self, state):
        """ Нужно для десериализации через pickle. """
        self.load_all(state)
