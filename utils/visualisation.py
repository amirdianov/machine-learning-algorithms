import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.metrics import MSE


class Visualisation:

    @staticmethod
    def visualise_best_models(models):
        x = []
        y = []
        accuracy_test = []
        for model in models:
            x.append(
                f"M1: {model[3]}; L1: {model[4]}; L2: {model[5]}"
            )
            y.append(f"{model[1]}")
            accuracy_test.append(model[2])
        dataframe = pd.DataFrame(
            data={"model": x, "accuracy_valid": y, "accuracy_test": accuracy_test}
        )
        fig = px.scatter(
            dataframe,
            x="model",
            y="accuracy_valid",
            hover_data={"accuracy_test"},
            title="Ten best models",
        ).update_layout(
            xaxis_title="Parameters M1, L1, L2",
            yaxis_title="Accuracy in validation",
        )
        fig.write_html(f"Ten_best_models.html")
        fig.show()
