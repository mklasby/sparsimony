from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from sparsimony.utils import view_tensor_as_neuron
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


class AblatedTileScorer(ABCScorer):

    @classmethod
    def score(
        cls,
        mask: torch.Tensor,
        score_override: torch.Tensor | None = None,
        tile_view: None | str | Tuple[int] = None,
        *args,
        **kwargs,
    ):
        # TODO: Use context manager for reshaping
        score_override = cls.init_score_override(mask, score_override)
        _reshape = False
        if tile_view is not None:
            _reshape = True
            _orig_shape = mask.shape
            mask = cls._reshape_t_as_view(mask, tile_view)
            score_override = cls._reshape_t_as_view(score_override, tile_view)
        ablated_tile_idx = torch.argwhere(
            torch.count_nonzero(mask, dim=-1) == 0
        ).flatten()
        score_override[ablated_tile_idx] = ScoreOverrides.INELIGIBLE
        if _reshape:
            mask = cls._reshape_t_as_view(mask, _orig_shape)
            score_override = cls._reshape_t_as_view(score_override, _orig_shape)
        return score_override

    @classmethod
    def _reshape_t_as_view(cls, t: torch.Tensor, view: str | Tuple[int]):
        if view == "neuron":
            return view_tensor_as_neuron(t)
        elif isinstance(view, Tuple):
            return t.view(view)
        else:
            raise NotImplementedError(f"Tile view {view} not supported!")


# MetaScorers
class TopKElementScorer(ABCScorer):
    def __init__(self, scorer: ABCScorer):
        self.scorer = scorer

    def score(
        self,
        k: int,
        largest: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        scores = self.scorer.score(*args, **kwargs)
        _, idx = torch.topk(scores.view(-1), k=k, dim=-1, largest=largest)
        scores = torch.scatter(
            torch.zeros_like(scores.view(-1)),
            dim=-1,
            index=idx,
            src=torch.ones_like(scores.view(-1)),
        ).view(scores.shape)
        return scores


class SequentialScorer(ABCScorer):
    def __init__(
        self,
        scorers: List[ABCScorer],
        agg_fn: Callable = torch.mean,
    ):
        self.scorers = scorers
        self.agg_fn = agg_fn

    def score(
        self,
        scorer_kwargs: List[Dict[Any, Any]],
        *args,
        **kwargs,
    ) -> List[torch.Tensor]:
        # TODO: Could return len(self.scores) dim tensor
        # TODO: make constraints on agg_func more clear
        scores = []
        for scorer, scorer_kwargs in list(zip(self.scorers, scorer_kwargs)):
            scores.append(scorer.score(**scorer_kwargs))
        return self.agg_fn(scores)


# class ScorerList(ABCScorer):
#     def __init__(self, scorers: List[ABCScorer]):
#         self.scorers = scorers

#     def score(self, *args, **kwargs):
#         for scorer in self.scorers:
#             scorer =

#         return super().score(*args, **kwargs)


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
