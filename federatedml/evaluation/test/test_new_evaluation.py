from federatedml.evaluation.backup.evaluation import Evaluation
from federatedml.evaluation.evaluation import EvaluateParam
from federatedml.evaluation.metrics.classification_metric \
    import BiClassAccuracy, BiClassRecall, BiClassPrecision, KS, Lift, Gain
from federatedml.evaluation.metric_interface import Metrics

import numpy as np

import time

scores = np.random.random(10000)
labels = (scores > 0.5) + 0

param = EvaluateParam(pos_label=1)
eval = Evaluation()
eval._init_model(param)

eval.eval_type = 'binary'
# rs = eval.recall(labels, scores)

s_p_0 = time.time()
# rs_1 = eval.precision(labels, pred_scores=scores)
e_p_0 = time.time()

# rs_2 = BiClassRecall().compute(labels, scores)

s_p_1 = time.time()
rs_2 = BiClassPrecision().compute(labels, scores)
e_p_1 = time.time()

print('precision old ver time {}, new ver time {}'.format(e_p_0 - s_p_0, e_p_1 - s_p_1))

s_r_0 = time.time()
# rs_3 = eval.recall(labels, pred_scores=scores)
e_r_0 = time.time()

# rs_2 = BiClassRecall().compute(labels, scores)

s_r_1 = time.time()
rs_4 = BiClassRecall().compute(labels, scores)
e_r_1 = time.time()

print('recall old ver time {}, new ver time {}'.format(e_r_0 - s_r_0, e_r_1 - s_r_1))

s_a_0 = time.time()
# rs_5 = eval.accuracy(labels, scores)
e_a_0 = time.time()

s_a_1 = time.time()
rs_6 = BiClassAccuracy().compute(labels, scores)
e_a_1 = time.time()

print('acc old ver time {}, new ver time {}'.format(e_a_0 - s_a_0, e_a_1 - s_a_1))

s_ks_0 = time.time()
# rs_7 = eval.ks(labels, scores)
e_ks_0 = time.time()

s_ks_1 = time.time()
rs_8 = KS.compute(labels, scores)
e_ks_1 = time.time()

print('ks old ver time {}, new ver time {}'.format(e_ks_0 - s_ks_0, e_ks_1 - s_ks_1))

s_l_0 = time.time()
# rs_9 = eval.lift(labels, scores)
e_l_0 = time.time()

s_l_1 = time.time()
rs_10 = Lift().compute(labels, scores)
e_l_1 = time.time()

print('lift old ver time {}, new ver time {}'.format(e_l_0 - s_l_0, e_l_1 - s_l_1))

s_g_0 = time.time()
# rs_11 = eval.gain(labels, scores)
e_g_0 = time.time()

s_g_1 = time.time()
rs_12 = Gain().compute(labels, scores)
e_g_1 = time.time()

print('gain old ver time {}, new ver time {}'.format(e_g_0 - s_g_0, e_g_1 - s_g_1))

metric_interface = Metrics(pos_label=1, eval_type='binary')



# print('start evaluating')
#
# s1 = time.time()
# rs_1 = eval.lift(labels, pred_scores=scores)
# e1 = time.time()
# print(e1 - s1)
#
#
# s0 = time.time()
# rs_0 = eval.lift_fast(labels, pred_scores=scores)
# e0 = time.time()
# print(e0 - s0)
#
# lift_x = np.array(rs_0[0])
# lift_y = np.array(rs_0[1])
#
# lift_x_ = np.array(rs_1[0])
# lift_y_ = np.array(rs_1[1])



# s0 = time.time()
# ks, fpr, tpr, thres, cuts = eval.ks(labels, scores)
# tpr = np.array(tpr)
# fpr = np.array(fpr)
# e0 = time.time()
# print(e0 - s0)
#
# s1 = time.time()
# # tpr_, fpr_, pos_num, neg_num, cuts_, thres_ = eval.compute_tpr_and_fpr(labels, scores)
# ks_val, fpr_, tpr_, thres_, cuts_ = eval.ks_fast(labels, scores)
# e1 = time.time()
# print(e1 - s1)