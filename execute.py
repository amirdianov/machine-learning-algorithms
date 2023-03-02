from configs.linear_regression_cfg import cfg as lin_reg_cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.metrics import MSE
from utils.visualisation import Visualisation


def experiment(lin_reg_cfg):
    debugger = False
    # lin_reg_model = LinearRegression(lin_reg_cfg.base_functions)
    linreg_dataset = LinRegDataset(lin_reg_cfg)

    # predictions = lin_reg_model(linreg_dataset["inputs"])
    # error = MSE(predictions, linreg_dataset["targets"])
    #
    # if debugger:
    #     Visualisation.visualise_predicted_trace(
    #         predictions,
    #         linreg_dataset["inputs"],
    #         linreg_dataset["targets"],
    #         plot_title=f"Полином степени {len(lin_reg_cfg.base_functions)}; MSE = {round(error, 2)}",
    #     )


if __name__ == "__main__":
    experiment(lin_reg_cfg)
