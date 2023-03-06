import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.metrics import MSE


class Visualisation:
    @staticmethod
    def visualise_predicted_trace(models: list, title=None) -> None:
        # visualise predicted trace and targets
        fig = go.Figure()
        for model in models:
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
            name = (
                "prediction without reg"
                if model[0].reg_coeff == 0
                else "prediction with reg"
            )
            trace2 = go.Scatter(
                x=input_test, y=prediction_test, mode="lines", name=name
            )
            fig.add_trace(trace2)
        fig.add_trace(trace1)
        fig.update_layout(
            title=f"{title}; Полиномы степени {100}; MSE: without reg {round(models[0][1], 2)}; with reg {round(models[1][1], 2)}"
        )
        fig.write_html(f"{title}.html")
        fig.show()

    @staticmethod
    def visualise_best_models(models):
        x = []
        y = []
        mse_test = []
        for model in models:
            x.append(
                f"degree: {len(model[0].base_functions)}; reg_coeff: {model[0].reg_coeff}"
            )
            y.append(f"{model[1]}")
            prediction_test = model[0](model[2])
            mse_test.append(MSE(prediction_test, model[3]))
        fig = px.scatter(
            x=x, y=y, hover_data=[mse_test], title="Ten best models"
        ).update_layout(
            xaxis_title="Max degree and regularisation coefficient",
            yaxis_title="MSE in validation",
        )
        # fig.write_html(f"Ten_best_models.html")
        fig.show()
