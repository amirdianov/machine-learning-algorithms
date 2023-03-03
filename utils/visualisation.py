import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Visualisation:
    @staticmethod
    def visualise_predicted_trace(model: tuple) -> None:
        # visualise predicted trace and targets
        fig = go.Figure()
        prediction_test = model[0](model[2])
        df = pd.DataFrame(np.array(model[2]))
        sample = pd.DataFrame(
            np.concatenate(
                (np.array(df), np.array(prediction_test).reshape(-1, 1)), axis=1
            ),
            columns=["input", "prediction"],
        ).sort_values("input")
        input_test = np.array(sample["input"])
        prediction_test = np.array(sample["prediction"])
        trace1 = go.Scatter(x=model[2], y=model[3], mode="markers", name="target")
        trace2 = go.Scatter(
            x=input_test, y=prediction_test, mode="lines", name="prediction"
        )
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.update_layout(
            title=f"Полином степени {len(model[0].base_functions)}; MSE = {round(model[1], 2)}"
        )
        fig.show()

    @staticmethod
    def visualise_best_models(*args, **kwargs):
        pass
