import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc


class Visualisation:
    @staticmethod
    def visualise_pr_curve(metrics):
        recall_arr = []
        precision_arr = []
        for data in metrics:
            recall_arr.append(data['recall'])
            precision_arr.append(data['precision'])
        ans = sorted(np.column_stack([np.array(recall_arr), np.array(precision_arr)]), key=lambda x: x[0],
                     reverse=False)
        ans.insert(0, [0, 0])
        ans.append([1, 0])
        df = pd.DataFrame(ans, columns=['recall', 'precision'])
        S = auc(np.array(df['recall']), np.array(df['precision']))
        for i in range(len(ans) - 1, 0, -1):
            ans[i - 1][1] = max(ans[i - 1][1],
                                ans[i][1])
        x = []
        y = []
        fig = go.Figure()
        for pair in ans:
            x.append(pair[0])
            y.append(pair[1])
        trace1 = go.Scatter(x=x, y=y, mode="lines", name="target")
        fig.add_trace(trace1)
        fig.show()
