import numpy


def calculate_classes_distribution(y, indices):
    """ Вычислить распределение классов в обучающей выборке по набору меток классов для примеров из этой выборки.

    :param y: метки классов для примеров обучающей выборки
    (одномерная последовательность целых - list, tuple или numpy.ndarray).
    :param indices: индексы тех примеров, чьи метки классов мы будем принимать во внимание (а другие пропускать).

    :return Словарь, ключами которого - целочисленные метки классов, а значения - количество примеров для этих классов.

    """
    classes_distribution = dict()
    for sample_ind in indices:
        class_ind = y[sample_ind]
        if class_ind in classes_distribution:
            classes_distribution[class_ind] += 1
        else:
            classes_distribution[class_ind] = 1
    return classes_distribution


def split_train_data(y, n_validation, random_state):
    """ Разбить обучающую выборку случайным образом на train и validation.

    Разбиение выполняется с использованием заданного генератора случайных чисел, но так, чтобы распределение классов
    в полученных подвыборках не сильно отличалось от распределения классов в исходной выборке.

    :param y: метки классов для примеров обучающей выборки
    (одномерная последовательность целых - list, tuple или numpy.ndarray).
    :param n_validation: сколько примеров мы заберём для validation.
    :param random_state: используемый генератор случайных чисел.

    :return два numpy.ndarray-массива, в 1-м из которых лежат индексы примеров для train, а во 2-м - для validation.

    """
    indices = numpy.arange(0, y.shape[0], 1, numpy.int32)
    classes_distribution = calculate_classes_distribution(y, indices)
    n_train = len(indices) - n_validation
    if (n_train < len(classes_distribution)) or (n_validation < len(classes_distribution)):
        raise ValueError('Train data cannot be split into train and validation subsets!')
    random_state.shuffle(indices)
    train_classes_distribution = calculate_classes_distribution(y, indices[:n_train])
    val_classes_distribution = calculate_classes_distribution(y, indices[n_train:])
    while (set(classes_distribution.keys()) != set(train_classes_distribution.keys())) or \
            (set(classes_distribution.keys()) != set(val_classes_distribution.keys())):
        random_state.shuffle(indices)
        train_classes_distribution = calculate_classes_distribution(y, indices[:n_train])
        val_classes_distribution = calculate_classes_distribution(y, indices[n_train:])
    return indices[:n_train], indices[n_train:]


def iterate(indices, batchsize, shuffle, random_state=None):
    """ Итерироваться по батчам из заданных примеров выборки данных (обучающей или просто входной, неважно).

    Каждый батч всегда имеет фиксированный размер batchsize. В ситуации, когда количество примеров выборки (т.е. размер
    массива indices) не делится нацело на batchsize, то в последний батч добавляются примеры из первого батча с тем,
    чтобы добрать недостающее количество примеров.

    Если требуется итерироваться не последовательно, а случайно (т.е. аргумент-флажок shuffle установлен в True), то
    перед началом итерирования все индексы indices случайным образом перемешиваются с помощью генератора случайных чисел
    random_state.

    :param indices: индексы примеров выборки, которые нам интересны и по которым мы хотим итерироваться.
    :param batchsize: размер батча.
    :param shuffle: будем ли мы итерироваться случайно или же по порядку.
    :param random_state: генератор случайных чисел, используемый в том случае, если мы хотим итерироваться случайно.

    :return итератор по батчам; каждый элемент - это numpy.ndarray-массив индексов примеров, попавших в этот батч.

    """
    if not isinstance(indices, numpy.ndarray):
        raise ValueError('`indices` must be a numpy.ndarray!')
    if len(indices.shape) != 1:
        raise ValueError('`indices` must be a 1-D array!')
    if (indices.dtype != numpy.int32) and (indices.dtype != numpy.uint64) and (indices.dtype != numpy.int32) and \
            (indices.dtype != numpy.uint32):
        raise ValueError('Items of the `indices` array must be integer numbers!')
    n_samples = indices.shape[0]
    if n_samples < 1:
        return
    n_batches = n_samples // batchsize
    if (n_batches * batchsize) < n_samples:
        n_batches += 1
    if shuffle:
        prepared_indices = indices.copy()
        random_state.shuffle(prepared_indices)
    else:
        prepared_indices = indices
    for batch_ind in range(n_batches):
        indices_in_batch = numpy.empty((batchsize,), dtype=numpy.int32)
        start_pos = batch_ind * batchsize
        for counter in range(batchsize):
            indices_in_batch[counter] = prepared_indices[start_pos]
            start_pos += 1
            if start_pos >= prepared_indices.shape[0]:
                start_pos = 0
        yield indices_in_batch
    del prepared_indices
