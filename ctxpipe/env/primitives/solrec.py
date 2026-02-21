"""SoluRec-oriented primitive implementations for CtxPipe operator-space switch."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

from .primitive import Primitive


def _split_num_cat(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = list(data._get_numeric_data().columns)
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    return data[cat_cols].copy(), data[num_cols].copy()


def _merge_cat_num(cat: pd.DataFrame, num: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([cat.reset_index(drop=True), num.reset_index(drop=True)], axis=1)


class ImputerConstantPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="ImputerConstant")
        self.id = 5
        self.gid = 30
        self.type = "ImputerNum"
        self.description = "Impute numeric columns with constant 0."
        self.imp = SimpleImputer(strategy="constant", fill_value=0.0)
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        return data.isnull().any().any()

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0:
            return train_x, test_x

        self.imp.fit(num_train)
        num_train = pd.DataFrame(
            self.imp.transform(num_train),
            columns=list(num_train.columns),
        ).infer_objects()
        num_test = pd.DataFrame(
            self.imp.transform(num_test),
            columns=list(num_test.columns),
        ).infer_objects()

        return _merge_cat_num(cat_train, num_train), _merge_cat_num(cat_test, num_test)


class ImputerKNNPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="ImputerKNN")
        self.id = 6
        self.gid = 31
        self.type = "ImputerNum"
        self.description = "Impute numeric columns with KNN."
        self.accept_type = "c"
        self.need_y = False
        self._imputer = None

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        return data.isnull().any().any()

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0:
            return train_x, test_x

        n_neighbors = min(5, max(1, len(num_train) - 1))
        if len(num_train) <= 1:
            self._imputer = SimpleImputer(strategy="mean")
        else:
            self._imputer = KNNImputer(n_neighbors=n_neighbors)

        self._imputer.fit(num_train)
        num_train = pd.DataFrame(
            self._imputer.transform(num_train),
            columns=list(num_train.columns),
        ).infer_objects()
        num_test = pd.DataFrame(
            self._imputer.transform(num_test),
            columns=list(num_test.columns),
        ).infer_objects()

        return _merge_cat_num(cat_train, num_train), _merge_cat_num(cat_test, num_test)


class _OutlierBasePrim(Primitive):
    def __init__(self, name: str, gid: int):
        super().__init__(name=name)
        self.gid = gid
        self.type = "FeaturePreprocessing"
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c1(data)

    def is_needed(self, data):
        return True

    @staticmethod
    def _restore(cat_train, num_train, cat_test, num_test):
        train = _merge_cat_num(cat_train, num_train)
        test = _merge_cat_num(cat_test, num_test)
        return train, test


class OutlierIQRClipPrim(_OutlierBasePrim):
    def __init__(self, random_state=0):
        super().__init__(name="OutlierIQR", gid=32)
        self.id = 30

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0:
            return train_x, test_x

        for col in num_train.columns:
            q1, q3 = num_train[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            num_train[col] = num_train[col].clip(lower=lower, upper=upper)
            num_test[col] = num_test[col].clip(lower=lower, upper=upper)

        return self._restore(cat_train, num_train, cat_test, num_test)


class OutlierZScoreClipPrim(_OutlierBasePrim):
    def __init__(self, random_state=0):
        super().__init__(name="OutlierZScore", gid=33)
        self.id = 31

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0:
            return train_x, test_x

        mean = num_train.mean(axis=0)
        std = num_train.std(axis=0).replace(0, np.nan)
        lower = mean - 3.0 * std
        upper = mean + 3.0 * std

        for col in num_train.columns:
            lo = lower[col] if pd.notna(lower[col]) else num_train[col].min()
            hi = upper[col] if pd.notna(upper[col]) else num_train[col].max()
            num_train[col] = num_train[col].clip(lower=lo, upper=hi)
            num_test[col] = num_test[col].clip(lower=lo, upper=hi)

        return self._restore(cat_train, num_train, cat_test, num_test)


class OutlierLOFCleanPrim(_OutlierBasePrim):
    def __init__(self, random_state=0):
        super().__init__(name="OutlierLOF", gid=34)
        self.id = 32
        self._model = None

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0 or len(num_train) < 3:
            return train_x, test_x

        med = num_train.median(axis=0)
        n_neighbors = min(20, max(2, len(num_train) - 1))
        self._model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        self._model.fit(num_train)

        train_mask = self._model.predict(num_train) == -1
        test_mask = self._model.predict(num_test) == -1

        if train_mask.any():
            num_train.loc[train_mask, :] = med.values
        if test_mask.any():
            num_test.loc[test_mask, :] = med.values

        return self._restore(cat_train, num_train, cat_test, num_test)


class OutlierIsolationForestCleanPrim(_OutlierBasePrim):
    def __init__(self, random_state=0):
        super().__init__(name="OutlierIsolationForest", gid=35)
        self.id = 33
        self._model = IsolationForest(contamination=0.05, random_state=42)

    def transform(self, train_x, test_x, train_y):
        cat_train, num_train = _split_num_cat(train_x)
        cat_test, num_test = _split_num_cat(test_x)
        if num_train.shape[1] == 0 or len(num_train) < 3:
            return train_x, test_x

        med = num_train.median(axis=0)
        self._model.fit(num_train)

        train_mask = self._model.predict(num_train) == -1
        test_mask = self._model.predict(num_test) == -1

        if train_mask.any():
            num_train.loc[train_mask, :] = med.values
        if test_mask.any():
            num_test.loc[test_mask, :] = med.values

        return self._restore(cat_train, num_train, cat_test, num_test)


class SelectKBestFClassifPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="SelectKBestFClassif")
        self.id = 34
        self.gid = 36
        self.type = "FeatureSelection"
        self.description = "Feature selection with ANOVA F-score."
        self.accept_type = "c_t"
        self.need_y = True
        self.selector = None
        self._final_cols = None

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return data.shape[1] > 1

    def transform(self, train_x, test_x, train_y):
        if train_x.shape[1] <= 1:
            return train_x, test_x

        k = min(20, train_x.shape[1])
        self.selector = SelectKBest(f_classif, k=k)
        self.selector.fit(train_x, np.asarray(train_y).ravel())
        mask = self.selector.get_support(indices=False)
        self._final_cols = list(pd.Index(train_x.columns)[mask])
        if not self._final_cols:
            return train_x, test_x

        train_out = pd.DataFrame(
            self.selector.transform(train_x),
            columns=self._final_cols,
        )
        test_out = pd.DataFrame(
            self.selector.transform(test_x),
            columns=self._final_cols,
        )
        return train_out, test_out


class SelectKBestMutualInfoPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="SelectKBestMutualInfo")
        self.id = 35
        self.gid = 37
        self.type = "FeatureSelection"
        self.description = "Feature selection with mutual information."
        self.accept_type = "c_t"
        self.need_y = True
        self.selector = None
        self._final_cols = None

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return data.shape[1] > 1

    def transform(self, train_x, test_x, train_y):
        if train_x.shape[1] <= 1:
            return train_x, test_x

        k = min(20, train_x.shape[1])
        self.selector = SelectKBest(
            lambda Xv, yv: mutual_info_classif(Xv, yv, discrete_features="auto"),
            k=k,
        )
        self.selector.fit(train_x, np.asarray(train_y).ravel())
        mask = self.selector.get_support(indices=False)
        self._final_cols = list(pd.Index(train_x.columns)[mask])
        if not self._final_cols:
            return train_x, test_x

        train_out = pd.DataFrame(
            self.selector.transform(train_x),
            columns=self._final_cols,
        )
        test_out = pd.DataFrame(
            self.selector.transform(test_x),
            columns=self._final_cols,
        )
        return train_out, test_out
