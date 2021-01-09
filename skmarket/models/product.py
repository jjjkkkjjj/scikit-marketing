from ._base import NonLinearRegression
ここから
class BassModel(NonLinearRegression):
    def __init__(self, p=None, q=None, M=None, mode='F_SMf'):
        super().__init__()
        self._p = _check_ins('p', p, (int, float), allow_none=True)
        self._q = _check_ins('q', q, (int, float), allow_none=True)
        self._M = _check_ins('M', M, int, allow_none=True)
        self._mode_list = ['F_SMf', 'Fsumf_f', 'FsumDEFtm1_DEf']
        if not mode in self._mode_list:
            raise ValueError('mode must be of {}, but got {}'.format(self._mode_list, mode))
        self._mode = mode

    @property
    def peak_t(self):
        if self._mode == 'F_SMf':
            return np.log(self._q / self._p) / (self._p + self._q) + 1
        elif self._mode == 'Fsumf_f':
            return np.log(self._q / self._p) / (self._p + self._q)
        else:
            raise ValueError('peak is defined under {}'.format(self._mode_list[:2]))

    def fit(self, y):
        """
        fitting with non linear regression
        =======:param X: time sequence ndarray, shape = (T,)=======
        :param y: sales ndarray given by X, shape = (T,)
        :return self: BassModel
        """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        # X = _check_ins('X', X, (list, tuple, np.ndarray))
        y = _check_ins('y', y, (list, tuple, np.ndarray))

        # _X = np.array(X)
        _y = np.array(y)

        assert _y.ndim == 1, 'Dimension must be 1, but got {}'.format(_y.ndim)

        if self._mode == 'F_SMf':
            _X = np.arange(1, _y.size + 1)
        else:
            _X = np.arange(_y.size)

        popt, pcov = curve_fit(a_root, _X, _y, p0=(0.1, 0.38, _y.sum()))

        self._p, self._q, self._M = popt

        return self

    def predict(self, T):
        """
        :param T: int or float, last time value  or array-like time sequence
        :returns
            adopters: ndarray adopters given by t, shape = (T,)
            total_adopters: ndarray total adopters given by t, shape = (T,)
        """
        _ = _check_ins('T', T, (list, tuple, np.ndarray, int, float))

        if isinstance(T, (np.ndarray, list, tuple)):
            t = np.array(T)

        if self._mode == 'F_SMf':
            if isinstance(T, (int, float)):
                t = np.arange(T + 1)

            F = F_root(t, self._p, self._q, self._M)
            f = np.ediff1d(F, to_begin=F[0])

            return self._M * f, self._M * F

        elif self._mode == 'Fsumf_f':
            if isinstance(T, (int, float)):
                t = np.arange(T)  # start from 1

            f = f_root(t, self._p, self._q, self._M)
            F = np.cumsum(f)

            return self._M * f, self._M * F

        elif self._mode == 'FsumDEFtm1_DEf':
            if isinstance(T, (int, float)):
                t = np.arange(T)  # start from 1

            def DE_a(Atm1):
                return p * self._M + (q - p) * Atm1 - (q / self._M) * (Atm1 ** 2)

            a = [0]
            A = [0]
            for _ in t:
                a += [DE_a(A[-1])]
                A += [a[-1] + A[-1]]

            a = a[1:]
            A = A[1:]
            return np.array(a), np.array(A)

        else:
            assert False, 'Invalid mode'

    @classmethod
    def F_root(self, t, p, q, M):
        """
        :param t: array-like time sequences, shape = (T,)
        :param p: number
        :param q: number
        :param M: number
        :return F: array-like F given by t, shape = (T,)
        """
        times = np.array(t)

        numerator = 1 - np.exp(-(p + q) * times)
        denominator = 1 + (q / p * np.exp(-(p + q) * times))

        return numerator / denominator

    @classmethod
    def f_root(self, t, p, q, M):
        """
        :param t: array-like time sequences, shape = (T,)
        :param p: number
        :param q: number
        :param M: number
        :return f: array-like f given by t, shape = (T,)
        """
        times = np.array(t)

        numerator = (p + q) ** 2 / p * np.exp(-(p + q) * times)
        denominator = (1 + (q / p * np.exp(-(p + q) * times))) ** 2

        return numerator / denominator

    @classmethod
    def A_root(self, t, p, q, M):
        """
        :param t: array-like time sequences, shape = (T,)
        :param p: number
        :param q: number
        :param M: number
        :return f: array-like f given by t, shape = (T,)
        """
        return BassModel.F_root(t, p, q, M) * M

    @classmethod
    def a_root(self, t, p, q, M):
        """
        :param t: array-like time sequences, shape = (T,)
        :param p: number
        :param q: number
        :param M: number
        :return f: array-like f given by t, shape = (T,)
        """
        return BassModel.f_root(t, p, q, M) * M


def _check_ins(name, val, cls, allow_none=False, default=None):
    if allow_none and val is None:
        return default

    if not isinstance(val, cls):
        if isinstance(cls, (tuple, list)):
            desired = [c.__name__ for c in cls]
        else:
            desired = cls.__name__

        raise ValueError('{} must be {}, but got {}'.format(name, desired, type(val).__name__))

    return val
