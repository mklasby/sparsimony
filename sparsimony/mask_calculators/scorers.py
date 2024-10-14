from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist


# TODO: Then UnstructuredMagnitudePruner + UnstrcturedRandomGrower then test SET


class ScoreOverrides:
    # TODO: Convert score_override into tensor subclass
    # TODO: Do we need a pass through sentinel?
    INACTIVE = 0.0
    ACTIVE = 1.0
    MAX_SCORE = float("inf")  # fill for inactive param scores
    MIN_SCORE = -float("inf")  # fill for active param scores
    PADDING = -42.0  # sentinel for padded tensors, currently unused
    INELIGIBLE = float("NaN")  # For skipping tiles entirely
    EPS = torch.finfo(torch.float).smallest_normal


class ABCScorer(ABC):

    @classmethod
    @abstractmethod
    def score(
        cls,
        values: torch.Tensor,
        *args,
        **kwargs,
    ): ...

    @classmethod
    def override_active_scores(
        cls,
        scores: torch.Tensor,
        score_override: torch.Tensor,
        fill_value: float = ScoreOverrides.MIN_SCORE,
    ) -> torch.Tensor:
        scores = torch.where(
            torch.logical_or(
                score_override == ScoreOverrides.ACTIVE,
                score_override == ScoreOverrides.INELIGIBLE,
            ),
            torch.full_like(scores, fill_value),
            scores,
        )
        return scores

    @classmethod
    def override_inactive_scores(
        cls,
        scores: torch.Tensor,
        score_override: torch.Tensor,
        fill_value: float = ScoreOverrides.MAX_SCORE,
    ) -> torch.Tensor:
        scores = torch.where(
            torch.logical_or(
                score_override == ScoreOverrides.INACTIVE,
                score_override == ScoreOverrides.INELIGIBLE,
            ),
            torch.full_like(scores, fill_value),
            scores,
        )
        return scores

    @classmethod
    def candidate_tiles(
        cls,
        score_override: torch.Tensor,
    ) -> torch.Tensor:
        """Returns tiles where at least one element is eligible for scoring.

        Args:
            score_override (torch.Tensor): Score override tensor

        Returns:
            torch.Tensor: Indices of eligible tiles corresponding with first
                dim of score_override.
        """
        return torch.argwhere(
            score_override != ScoreOverrides.INELIGIBLE,
        )[:, 0].unique()

    @classmethod
    def all_reduce_scores(cls, scores: torch.Tensor) -> None:
        if dist.is_initialized() and scores.device != torch.device("cpu"):
            # For metrics such as weight magnitude, we don't need to all_reduce.
            # However, hard to guarantee so generally we perform all_reduce
            # except for in the case where scores are on CPU
            # (only supports gloo backend)
            dist.all_reduce(scores, dist.ReduceOp.AVG, async_op=False)

    @classmethod
    def init_score_override(
        cls,
        mask: torch.Tensor,
        score_override: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if score_override is None:
            score_override = torch.clone(mask).detach().to(dtype=torch.float)
        score_override = torch.where(
            score_override != ScoreOverrides.INELIGIBLE,
            mask,
            score_override,
        )
        return score_override


class MagnitudeScorer(ABCScorer):

    @classmethod
    def score(
        cls,
        values: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.abs(values)


class RandomScorer(ABCScorer):

    @classmethod
    def score(cls, values: torch.Tensor, *args, **kwargs):
        return torch.abs(torch.rand_like(values)) + ScoreOverrides.EPS


# class ThresholdPruner(BasePruner):
#     @classmethod
#     def get_scores(
#         cls,
#         mask: torch.Tensor,
#         candidate_tiles: torch.Tensor,
#         scorer: ABCMaskCalculator,
#         scoring_tensor: torch.Tensor,
#         score_threshold: float,
#         *args,
#         **kwargs,
#     ):
#         scores = scorer.get_scores(mask, candidate_tiles, scoring_tensor)
#         return torch.where(
#             scores > score_threshold,
#             torch.one_like(mask),
#             torch.zeros_like(mask),
#         )


# class RandomGrower(BaseGrower):
#     @classmethod
#     def get_scores(
#         cls, mask: torch.Tensor, candidate_tiles: torch.Tensor, *args,
# **kwargs
#     ):
#         return torch.where(
#             torch.logical_and(mask == 0, mask != cls._SCORE_FILL_VALUE),
#             torch.abs(torch.rand_like(mask))
#             + cls._EPS,  # small eps for avoiding 0s
#             torch.full_like(mask, cls._SCORE_FILL_VALUE),
#         )


# class GradientGrower(BaseGrower):

#     @classmethod
#     def get_scores(
#         cls,
#         mask: torch.Tensor,
#         candidate_tiles: torch.Tensor,
#         grads: torch.Tensor | None,
#         *args,
#         **kwargs,
#     ):
#         if grads is None:
#             # Randomly grow
#             return RandomGrower.get_scores(mask, candidate_tiles)
#         return torch.where(
#             torch.logical_and(mask == 0, mask != cls._SCORE_FILL_VALUE),
#             torch.abs(grads[candidate_tiles]),
#             torch.full_like(mask, cls._SCORE_FILL_VALUE),
#         )
