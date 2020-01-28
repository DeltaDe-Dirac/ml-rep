import numpy as np
import scipy.special as sc


def sc_softmax(in_logits):
    return sc.softmax(in_logits)


def np_softmax(in_logits, round_digits=16):
    exponents = [np.exp(logit) for logit in in_logits]
    exponents_sum = np.sum(exponents, axis=0)
    return [np.round(i / exponents_sum, round_digits) for i in exponents]


def test_softmax(in_softmax):
    res_sum = 0
    for i in in_softmax:
        res_sum += i
    return res_sum


# ------------------------------- MAIN ------------------------------- #
if __name__ == '__main__':
    logits = [2.0, 1.0, .1]
    print(np_softmax(logits, round_digits=16))
    print(test_softmax(np_softmax(logits)))

    print(sc_softmax(logits))
    print(test_softmax(sc_softmax(logits)))
# ------------------------------- MAIN ------------------------------- #
