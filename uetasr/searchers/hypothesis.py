from tensorflow import Tensor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Optional[Tuple[Tensor, Optional[Tensor]]]
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
