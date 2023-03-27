from .gain import Gain
from .noise import GaussianNoise
from .speed import TimeStretch
from .pitch import PitchShift
try:
    from .telephony import Telephony
except ImportError:
    pass


__all__ = [
    "Gain",
    "GaussianNoise",
    "TimeStretch",
    "PitchShift",
    "Telephony"
]
