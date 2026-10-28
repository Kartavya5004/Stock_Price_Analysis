class NoDataError(ValueError):
    """The data provider returned nothing for this ticker (unknown symbol / delisted / outage)."""


class InsufficientDataError(ValueError):
    """Data exists but is too short to compute indicators, or to train/evaluate sensibly."""
