from collections import OrderedDict

import numpy as np
import pandas as pd

import extern.metrics as metrics


class Metrics:
    def __init__(self, label_set):
        self.label_set = label_set

    def scores(self, y_true, y_pred):
        df_dict = self.evaluate(y_true, y_pred)
        auprc_macro = metrics.macro_averaged_auprc(df_dict)
        # auprc_micro, eval_df = metrics.micro_averaged_auprc(df_dict, return_df=True)
        # f1_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]

        return OrderedDict({
            'auprc_macro': auprc_macro,
            # 'auprc_micro': auprc_micro,
            # 'f1_micro': eval_df["F"][f1_idx],
        })

    def evaluate(self, y_true, y_pred):
        df_dict = {}

        for k in range(1, 9):
            indexes = [index for index, label in enumerate(self.label_set)
                       if int(label[0]) == k and label[2] != 'X']
            thresholds = np.sort(np.ravel(np.copy(y_pred[:, indexes])))
            thresholds = thresholds[np.searchsorted(thresholds, 0.01):]
            thresholds = np.append(thresholds, 1.0)
            thresholds = np.unique(thresholds)[::-1]

            index_incomplete = indexes[-1] + 1
            has_incomplete = index_incomplete < y_true.shape[1] \
                and self.label_set[index_incomplete][2] == 'X'
            if has_incomplete:
                y_true_incomplete = y_true[:, index_incomplete]
            else:
                y_true_incomplete = np.zeros(len(y_true))

            TPs = np.zeros(len(thresholds), dtype=int)
            FPs = np.zeros(len(thresholds), dtype=int)
            FNs = np.zeros(len(thresholds), dtype=int)
            for i, t in enumerate(thresholds):
                y_pred_b = (y_pred[:, indexes] >= t).astype(int)

                if has_incomplete:
                    y_pred_incomplete = (y_pred[:, index_incomplete] >= t)
                else:
                    y_pred_incomplete = np.zeros(len(y_pred))

                TPs[i], FPs[i], FNs[i] = metrics.confusion_matrix_fine(
                    y_true[:, indexes], y_pred_b,
                    y_true_incomplete, y_pred_incomplete)

            eval_df = pd.DataFrame({
                "threshold": thresholds,
                "TP": TPs,
                "FP": FPs,
                "FN": FNs
            })

            eps = 1e-1  # Any eps < 1 suffices
            eval_df["P"] = TPs / np.maximum(TPs + FPs, eps)
            eval_df["R"] = TPs / np.maximum(TPs + FNs, eps)
            eval_df["F"] = 2 / (1/eval_df["P"] + 1/eval_df["R"])

            df_dict[k] = eval_df

        return df_dict
