# TODO: Update
# class HierarchicalMaskCalculator(ABCMaskCalculator):

#     @classmethod
#     def calculate_mask(
#         cls,
#         sparsities: List[float],
#         mask: torch.Tensor,
#         calculators: List[ABCMaskCalculator],
#         calculator_kwargs: List[Dict[Any, Any]],
#         score_override: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         # TODO: Maybe just for pruners?
#         assert len(sparsities) == len(calculators) == len(calculator_kwargs)
#         for sparsity, calculator, calc_kwargs in list(
#             zip(sparsities, calculators, calculator_kwargs)
#         ):
#             mask = calculator.calculate_mask(
#                 sparsity=sparsity,
#                 mask=mask,
#                 score_override=score_override,
#                 **calc_kwargs,
#             )
#             score_override = cls._get_score_override(
#                 mask,
#                 calculator._TILE_VIEW,
#                 score_override,
#             )
#         return mask

#     @classmethod
#     def _get_score_override(
#         cls,
#         mask: torch.Tensor,
#         tile_view: str | Tuple[int],
#         score_override: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         if score_override is None:
#             score_override = torch.zeros_like(mask)
#         _orig_shape = mask.shape
#         _reshape = True
#         if isinstance(tile_view, tuple) and mask.shape[-1] % tile_view[-1] != 0:  # noqa
#             cls._logger.warning(
#                 "Score override requires padding, will "
#                 "calculate override without reshaping mask"
#             )
#             _reshape = False
#         else:
#             mask = cls._reshape_t_as_view(mask, tile_view)
#             score_override = cls._reshape_t_as_view(score_override, tile_view)
#         ablated_tile_idx = torch.argwhere(
#             torch.count_nonzero(mask, dim=-1) == 0
#         ).flatten()
#         score_override[ablated_tile_idx] = cls._OVERRIDE_SENTINEL_VALUE
#         if _reshape:
#             mask = cls._reshape_t_as_view(mask, _orig_shape)
#             score_override = cls._reshape_t_as_view(score_override, _orig_shape)  # noqa
#         return score_override

#     @classmethod
#     def _reshape_t_as_view(cls, t: torch.Tensor, view: str | Tuple[int]):
#         if view == "neuron":
#             return view_tensor_as_neuron(t)
#         elif isinstance(view, Tuple):
#             return t.view(view)
#         else:
#             raise NotImplementedError(f"Tile view {view} not supported!")
