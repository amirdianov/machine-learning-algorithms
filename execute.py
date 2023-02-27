from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.metrics import MSE
from utils.visualisation import Visualisation


def experiment(lin_reg_cfg, visualise_prediction=True):
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions)
    linreg_dataset = LinRegDataset()(lin_reg_cfg.dataframe_path)

    predictions = lin_reg_model(linreg_dataset["inputs"])
    error = MSE(predictions, linreg_dataset["targets"])

    if visualise_prediction:
        Visualisation.visualise_predicted_trace(
            predictions,
            linreg_dataset["inputs"],
            linreg_dataset["targets"],
            plot_title=f"Полином степени {len(lin_reg_cfg.base_functions)}; MSE = {error}",
        )


if __name__ == "__main__":
    from configs.linear_regression_cfg import cfg as lin_reg_cfg

    degrees = [1, 8, 100]
    for elem in degrees:
        lin_reg_cfg.update(
            base_functions=[lambda x, degree=i: x**degree for i in range(1 + elem)]
        )
        experiment(lin_reg_cfg, visualise_prediction=True)
