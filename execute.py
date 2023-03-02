import numpy as np

from configs.linear_regression_cfg import cfg as lin_reg_cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression


def experiment(lin_reg_cfg, reg_coeff):
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, reg_coeff)
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
    count_models_to_train = 100
    polynomial_choose = [5, 200]
    reg_coeff_choose = [0, 5]
    for _ in range(count_models_to_train):
        polynimial = np.random.randint(polynomial_choose[0], polynomial_choose[1])
        reg_coeff = np.random.uniform(reg_coeff_choose[0], reg_coeff_choose[1])
        experiment(
            lin_reg_cfg.update(
                base_functions=[
                    lambda x, degree=i: x**degree for i in range(1, 1 + polynimial)
                ]
            ),
            reg_coeff,
        )
