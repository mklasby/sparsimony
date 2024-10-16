from typing import List, Dict, Optional, Any

import torch

from .base import ABCMaskCalculator
from .scorers import AblatedTileScorer


class HierarchicalMaskCalculator(ABCMaskCalculator):
    # TODO: Make stateful with
    @classmethod
    def calculate_mask(
        cls,
        sparsities: List[float],
        mask: torch.Tensor,
        calculators: List[ABCMaskCalculator],
        calculator_kwargs: List[Dict[Any, Any]],
        score_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Maybe just for pruners?
        assert len(sparsities) == len(calculators) == len(calculator_kwargs)
        for sparsity, calculator, calc_kwargs in list(
            zip(sparsities, calculators, calculator_kwargs)
        ):
            mask = calculator.calculate_mask(
                sparsity=sparsity,
                mask=mask,
                score_override=score_override,
                **calc_kwargs,
            )
            score_override = AblatedTileScorer.score(
                mask,
                score_override,
                calculator._TILE_VIEW,
            )
        return mask
