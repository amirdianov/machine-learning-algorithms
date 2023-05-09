import pandas as pd
import plotly.express as px


class Visualisation:

    @staticmethod
    def visualise_best_models(models):
        x = []
        y = []
        MSE_test = []
        for model in models:
            x.append(
                f"M: {model[3]}; a: {model[4]};"
            )
            y.append(f"{model[1]}")
            MSE_test.append(model[2])
        dataframe = pd.DataFrame(
            data={"model": x, "MSE_valid": y, "MSE_test": MSE_test}
        )
        fig = px.scatter(
            dataframe,
            x="model",
            y="MSE_valid",
            hover_data={"MSE_test"},
            title="Ten best models",
        ).update_layout(
            xaxis_title="Parameters M, a",
            yaxis_title="MSE in validation",
        )
        fig.write_html(f"Ten_best_models.html")
        fig.show()
