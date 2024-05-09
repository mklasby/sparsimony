import logging


class DenseNoOp(object):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._logger.warning("Training dense, no DST selected!")

    def prepare(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass
