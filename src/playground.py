from .core.cf import similarity_matrix
from .core.train import train_cf
from .io import (
    aggregate_all,
    aggregate_cross_validation,
    read_entries,
    read_matrix,
    read_split_entries,
    write_matrix,
)
from .loss import loss_mae
from .predictors.item_based_cf import item_based_cf
from .predictors.slope_one_cf import slope_one_cf
from .predictors.user_based_cf import user_based_cf
from .presets import dynamic_presets, presets
from .typing import Similarity
from .utils import round_prediction


def test_user_based_cf() -> None:
    name = 'ex2.txt'
    train_arr = read_entries('data/demo/train.' + name)
    test_arr = read_entries('data/demo/test.' + name)

    r, a, q = aggregate_all(train_arr, test_arr)
    conf = presets['corr'] + {'knn_k': 2}
    print(conf)
    predictions = user_based_cf(r, a, q, conf)
    print(predictions)
    q.take_answers(predictions, force_update=True, validate=False)
    # q.take_answers(user_based_cf(r, a, q, conf), force_update=True)
    print(q)


def test_item_based_cf():
    name = 'ex1.txt'
    train_arr = read_entries('data/demo/train.' + name)
    test_arr = read_entries('data/demo/test.' + name)

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


def test_slope_one():
    train_arr = read_entries('data/demo/train.so.txt')
    test_arr = read_entries('data/demo/test.so.txt')
    r, a, q = aggregate_cross_validation(train_arr, test_arr)
    print(r.raw)
    print(a.raw)
    print(q)
    conf = presets['slope_one'] + presets['case_amp']
    print(conf)
    predictions = slope_one_cf(r, a, q, conf)
    print(predictions)


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
    # conf = presets['item_based_k'] + presets[
    #     'adj_cos']    # * dynamic_presets['case_amp'](2.5)
    # conf = presets['slope_one']
    conf = presets['adj_cos'] + presets['item_based_k']
    print(conf)
    # sim_m = read_similarity(fname)
    predictions = item_based_cf(r, a, q, conf)
    # predictions = slope_one_cf(r, a, q, conf)
    # q.take_answers(predictions, force_update=True, validate=False)
    # q.take_answers(user_based_cf(r, a, q, conf), force_update=True)
    # print(predictions)
    # print(q._answers)
    print(
        loss_mae(q.ground_truth(),
                 [round_prediction(pred) for pred in predictions]))


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
    write_matrix(fname, Similarity(sim_m.raw))
    new_sim = read_matrix(fname)
    print(new_sim.raw)


def train():
    train_arr, test_arr = read_split_entries(0.05)
    r, a, q = aggregate_cross_validation(train_arr, test_arr)
    conf_list = []
    for k in range(1, 50):
        conf = (presets['corr'] + {
            'knn_k': k
        }) * dynamic_presets['case_amp'](2.5) + dynamic_presets['iuf'](r.raw)
        conf_list.append(conf)
    print(conf_list[0])
    train_cf(r, a, q, user_based_cf, conf_list)


if __name__ == '__main__':
    test()
