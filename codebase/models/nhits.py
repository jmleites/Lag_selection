from neuralforecast.models import NHITS


class NamedNHITS(NHITS):

    def __repr__(self):
        return f'NHITS_{self.input_size}'
