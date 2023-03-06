import numpy as np

from configs.linear_regression_cfg import cfg as lin_reg_cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.metrics import MSE
from utils.visualisation import Visualisation


def experiment(lin_reg_cfg, reg_coeff, data=None):
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, reg_coeff)
    if data is None:
        linreg_dataset = LinRegDataset(lin_reg_cfg)()
    else:
        linreg_dataset = data
    lin_reg_model.train_model(
        linreg_dataset["inputs"]["train"], linreg_dataset["targets"]["train"]
    )
    predictions_valid = lin_reg_model(linreg_dataset["inputs"]["valid"])
    error_valid = MSE(predictions_valid, linreg_dataset["targets"]["valid"])

    return (
        lin_reg_model,
        error_valid,
        linreg_dataset["inputs"]["test"],
        linreg_dataset["targets"]["test"],
    )


if __name__ == "__main__":
    min_count_models_to_train = 100
    polynomial_choose = [5, 200]
    reg_coeff_choose = [0, 5]
    models = []
    for _ in range(min_count_models_to_train):
        polynimial = np.random.randint(polynomial_choose[0], polynomial_choose[1])
        reg_coeff = np.random.uniform(reg_coeff_choose[0], reg_coeff_choose[1])
        lin_reg_cfg.update(
            base_functions=[
                lambda x, degree=i: x**degree for i in range(1, 1 + polynimial)
            ]
        )
        models.append(experiment(lin_reg_cfg, reg_coeff))
    models = sorted(models, key=lambda x: x[1])
    models_best = models[:10]
    Visualisation.visualise_best_models(models_best)
    # for model in models_best[:3]:
    #     Visualisation.visualise_predicted_trace(model)
    lin_reg_cfg.update(
        base_functions=[lambda x, degree=i: x**degree for i in range(1, 1 + 100)]
    )
    dataset = LinRegDataset(lin_reg_cfg)()
    model_no_reg = experiment(lin_reg_cfg, 0, dataset)
    model_reg = experiment(lin_reg_cfg, 1 * 10 ** (-5), dataset)
    Visualisation.visualise_predicted_trace([model_no_reg, model_reg], "Comparison")
