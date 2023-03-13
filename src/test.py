from .core.knn import similarity_matrix
from .core.shared.similarity import support_matrix
from .io import (
    aggregate_all,
    aggregate_cross_validation,
    read_entries,
    read_similarity,
    read_split_entries,
    write_similarity,
)
from .loss import loss_mae
from .predictors.item_based_cf import item_based_cf
from .predictors.user_based_cf import user_based_cf
from .presets import dynamic_presets, presets
from .typing import Similarity


def test_user_based_cf() -> None:
    name = 'ex2.txt'
    train_arr = read_entries('data/train.' + name)
    test_arr = read_entries('data/test.' + name)

    r, a, q = aggregate_all(train_arr, test_arr)
    # print(r.raw)
    # print(a.raw)
    # print(q.questions)
    print('--------------')
    print(support_matrix(a.raw, r.raw))
    print('--------------')
    conf = presets['corr'] + {'knn_k': 2}
    print(conf)
    predictions = user_based_cf(r, a, q, conf)
    print(predictions)
    q.take_answers(predictions, force_update=True, validate=False)
    # q.take_answers(user_based_cf(r, a, q, conf), force_update=True)
    print(q)


def test_item_based_cf():
    name = 'ex1.txt'
    train_arr = read_entries('data/train.' + name)
    test_arr = read_entries('data/test.' + name)

    r, a, q = aggregate_all(train_arr, test_arr)
    print(r.raw)
    print(a.raw)
    print(q.questions)
    print('--------------')
    conf = presets['item_based_k'] + presets['adj_cos']
    print(conf)
    predictions = item_based_cf(r, a, q, conf)
    print(predictions)
    q.take_answers(predictions, force_update=True, validate=False)
    # q.take_answers(user_based_cf(r, a, q, conf), force_update=True)
    print(q)


def test():
    # fname = 'models/knn/adj_cos/train0.05_item.txt'
    # fname = 'models/knn/cos/train5_user.txt'
    train_arr, test_arr = read_split_entries(0.05)
    r, a, q = aggregate_cross_validation(train_arr, test_arr)
    # print(a.raw)
    # print(q)
    # conf = presets['item_based']
    # print(conf)
    # k = 20
    conf = presets['item_based_k'] + presets[
        'adj_cos']    # * dynamic_presets['case_amp'](2.5)
    print(conf)
    # sim_m = read_similarity(fname)
    predictions = item_based_cf(r, a, q, conf, None)
    # q.take_answers(predictions, force_update=True, validate=False)
    # q.take_answers(user_based_cf(r, a, q, conf), force_update=True)
    # print(predictions)
    # print(q._answers)
    print(loss_mae(q._answers, predictions))


def dump_sim():
    fname = 'models/knn/adj_cos/train0.05_item.txt'
    train_arr, test_arr = read_split_entries(0.05)
    r, a, _ = aggregate_cross_validation(train_arr, test_arr)

    conf = presets['adj_cos']
    print(conf)
    s = a + r
    sim_m = similarity_matrix(s.raw.T, s.raw.T, conf.sim_scheme,
                              conf.sim_fill_value)
    print(sim_m.raw)
    print('------------------')
    write_similarity(fname, Similarity(sim_m.raw))
    new_sim = read_similarity(fname)
    print(new_sim.raw)


if __name__ == '__main__':
    test_user_based_cf()
