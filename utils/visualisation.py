import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.graph_objs import Layout


class Visualisation:
    @staticmethod
    def visualise_predicted_trace(
        prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=""
    ):
        # visualise predicted trace and targets
        """

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()
        trace1 = go.Scatter(x=inputs, y=targets, mode="markers", name="target")
        trace2 = go.Scatter(x=inputs, y=prediction, mode="lines", name="prediction")
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.update_layout(title=f"{plot_title}")
        fig.show()
        # plotly.offline.plot(
        #     {"data": [trace1, trace2], "layout": Layout(title=plot_title)}
        # )

    @staticmethod
    def visualise_error():
        pass
