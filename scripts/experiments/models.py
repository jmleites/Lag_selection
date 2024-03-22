from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive, WindowAverage
from statsforecast import StatsForecast
from neuralforecast.losses.pytorch import SMAPE
def initialize_models(data_name, freq_int, freq_str, input_list, models, horizon, df):
    stats_models = [
        SeasonalNaive(season_length=freq_int),
        WindowAverage(window_size=freq_int),
    ]

    if data_name == 'M3':
        sf = StatsForecast(
            models=stats_models,
            freq=freq_str,
            n_jobs=1,
        )
    else:
        sf = StatsForecast(
            models=stats_models,
            freq=1,
            n_jobs=1,
        )

    nhits_config = {
        'max_steps': 1000,
        'val_check_steps': 30,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 25,
        'start_padding_enabled': True,
        'valid_loss': SMAPE(),
        'accelerator': 'cpu'
    }

    for i in input_list:
        models.append(NHITS(h=horizon,
                            input_size=i,
                            **nhits_config))

    nf = NeuralForecast(models=models, freq=freq_str)

    return df, sf, nf