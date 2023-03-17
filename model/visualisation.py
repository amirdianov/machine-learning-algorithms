import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import auc


class Visualisation:
    @staticmethod
    def visualise_roc_curve(metrics):
        a_arr, b_arr = [], []
        for data in metrics:
            a_arr.append(data['a'])
            b_arr.append(1 - data['b'])
        ans = sorted(np.column_stack([np.array(a_arr), np.array(b_arr)]), key=lambda x: x[0], reverse=False)
        df = pd.DataFrame(ans, columns=['a', 'b'])
        S = auc(np.array(df['a']), np.array(df['b']))
        fig = px.line(
            df,
            x="a",
            y="b",
            title=f'ROC curve with AUC: {S}',
        ).update_layout(
            xaxis_title="a",
            yaxis_title="b-1",
        )
        fig.write_html(f"ROC_curve.html")
        fig.show()
    @staticmethod
    def visualise_pr_curve(metrics):
        recall_arr, precision_arr, accuracy_arr, f1_score_arr, threshold_arr = [], [], [], [], []
        for data in metrics:
            recall_arr.append(data['recall'])
            precision_arr.append(data['precision'])
            accuracy_arr.append(data['accuracy'])
            f1_score_arr.append(data['f1_score'])
            threshold_arr.append(data['threshold'])
        ans = sorted(np.column_stack(
            [np.array(recall_arr), np.array(precision_arr), np.array(accuracy_arr), np.array(f1_score_arr),
             np.array(threshold_arr)]),
            key=lambda x: x[0],
            reverse=False)
        ans.insert(0, [0, 0, None, None, None])
        ans.append([1, 0, None, None, None])
        for i in range(len(ans) - 1, 0, -1):
            ans[i - 1][1] = max(ans[i - 1][1],
                                ans[i][1])
        df = pd.DataFrame(ans, columns=['recall', 'precision', 'accuracy', 'f1_score', 'threshold'])
        S = auc(np.array(df['recall']), np.array(df['precision']))
        # x = []
        # y = []
        # fig = go.Figure()
        # for pair in ans:
        #     x.append(pair[0])
        #     y.append(pair[1])
        # trace1 = go.Scatter(x=x, y=y, mode="lines", name="target")
        # fig.add_trace(trace1)
        fig = px.line(
            df,
            x="recall",
            y="precision",
            hover_data={"accuracy", "f1_score", "threshold"},
            title=f'PR curve with AP: {S}',
        ).update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        fig.write_html(f"PR_curve.html")
        fig.show()
