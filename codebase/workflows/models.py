from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive, WindowAverage
from statsforecast import StatsForecast
from neuralforecast.losses.pytorch import SMAPE

from codebase.models.nhits import NamedNHITS

NHITS_CONFIG = {
    'max_steps': 1500,
    'val_check_steps': 50,
    'enable_checkpointing': True,
    'start_padding_enabled': True,
    'early_stop_patience_steps': 50,
    'valid_loss': SMAPE(),
    'accelerator': 'cpu'
}


def initialize_models(data_name, freq_int, freq_str, input_list, horizon, df):
    stats_models = [
        SeasonalNaive(season_length=freq_int),
        WindowAverage(window_size=freq_int),
    ]

    if data_name == 'M4':
        sf = StatsForecast(
            models=stats_models,
            freq=freq_int,
            n_jobs=1,
        )
    else:
        sf = StatsForecast(
            models=stats_models,
            freq=freq_str,
            n_jobs=1,
        )

    models = []
    for i in input_list:
        models.append(NamedNHITS(h=horizon, input_size=i, **NHITS_CONFIG))

    nf = NeuralForecast(models=models, freq=freq_str)

    return df, sf, nf
