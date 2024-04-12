from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import re
import logging
import numpy as np
import torch.nn as nn

from sparsimony.utils import get_original_tensor


class BaseDistribution(ABC):

    def __init__(
        self,
        skip_first_layer: bool = False,
        skip_last_layer: bool = False,
        excluded_types: Optional[List[type]] = None,
        excluded_mod_name_regexs: Optional[List[type]] = None,
    ):
        self.skip_first_layer = skip_first_layer
        self.skip_last_layer = skip_last_layer
        if excluded_types is None:
            excluded_types = []
        self.excluded_types = excluded_types
        if excluded_mod_name_regexs is None:
            excluded_mod_name_regexs = []
        self.excluded_mod_name_regexs = excluded_mod_name_regexs
        self._logger = logging.getLogger(__name__)
        self._cache: Dict[float, List[float]] = dict()

    @abstractmethod
    def __call__(
        self, sparsity: float, groups: List[Dict[str, Any]], *args, **kwargs
    ) -> List[Dict[str, Any]]: ...

    def _should_exclude(self, mod: nn.Module, name: str) -> bool:
        if type(mod) in self.excluded_types:
            return True
        for pattern in self.excluded_mod_name_regexs:
            if re.match(pattern, name):
                return True
        return False

    def _get_layer_el(self, layer_config: Dict[str, Any]) -> int:
        return getattr(
            layer_config["module"], layer_config["tensor_name"]
        ).numel()

    def _validate(self, sparsity) -> None:
        if sparsity > 1:
            raise ValueError(f"Adjusted sparsity > 1 in {self.__name__}")

    def _cache_loader(self, sparsity: float, groups: List[Dict[str, Any]]):
        if sparsity in self._cache:
            for config, cached_sparsity in list(
                zip(groups, self._cache[sparsity])
            ):
                config["sparsity"] = cached_sparsity
            return groups
        return None


class UniformDistribution(BaseDistribution):
    def __init__(
        self,
        skip_first_layer: bool = False,
        skip_last_layer: bool = False,
        excluded_types: Optional[List[type]] = None,
        excluded_mod_name_regexs: Optional[List[type]] = None,
    ):
        super().__init__(
            skip_first_layer,
            skip_last_layer,
            excluded_types,
            excluded_mod_name_regexs,
        )

    def __call__(
        self, sparsity: float, groups: List[Dict[str, Any]], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        if sparsity in self._cache:
            return self._cache_loader(sparsity, groups)
        dense_el, sparse_el, num_el = 0, 0, 0
        keep_dense = []
        for layer_config in groups:
            if self._should_exclude(
                layer_config["module"], layer_config["module_fqn"]
            ):
                keep_dense.append(True)
            else:
                keep_dense.append(False)
            layer_el = self._get_layer_el(layer_config)
            if keep_dense[-1]:
                dense_el += layer_el
            else:
                sparse_el += layer_el
            num_el += layer_el
        adj_sparsity = ((sparsity * num_el) - dense_el) / sparse_el
        self._validate(adj_sparsity)

        for dense, layer_config in list(zip(keep_dense, groups)):
            layer_config["sparsity"] = adj_sparsity if not dense else 0
        return groups


class ERKDistribution(BaseDistribution):
    def __init__(
        self,
        skip_first_layer: bool = False,
        skip_last_layer: bool = False,
        excluded_types: Optional[List[type]] = None,
        excluded_mod_name_regexs: Optional[List[type]] = None,
        erk_power_scale: float = 1.0,
    ):
        super().__init__(
            skip_first_layer,
            skip_last_layer,
            excluded_types,
            excluded_mod_name_regexs,
        )
        self.erk_power_scale = erk_power_scale

    def __call__(
        self, sparsity: float, groups: List[Dict[str, Any]], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get Erdos Renyi Kernel sparsity distribution for `self.model`.

        Implementation based on approach in original rigl paper and reproduced
        papers:
        https://github.com/google-research/rigl/blob/97d62b0724c9a489a5318edb34951c6800575311/rigl/sparse_utils.py#L90
        https://github.com/varun19299/rigl-reproducibility/blob/f8a3398f6249e291aa8d91e376e49820fde8f2d3/sparselearning/funcs/init_scheme.py#L147


        Returns:
            List[Dict[str, Any]: Modified group configs with sparsity added.
        """
        if sparsity in self._cache:
            return self._cache_loader(sparsity, groups)

        eps = None
        is_eps_valid = False
        dense_layers = set()

        for layer_idx, layer_config in enumerate(groups):
            if self._should_exclude(
                layer_config["module"], layer_config["module_fqn"]
            ):
                dense_layers.add(layer_idx)

        while not is_eps_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}

            for layer_idx, layer_config in enumerate(groups):
                weights = get_original_tensor(**layer_config)
                n_params = np.prod(weights.shape)  # Total number of params
                n_zeros = int(n_params * sparsity)
                n_ones = int(n_params * (1 - sparsity))

                if layer_idx in dense_layers:
                    # dense_layers.add(layer_idx)
                    rhs -= n_zeros
                else:
                    n_ones = n_params - n_zeros
                    rhs += n_ones
                    raw_prob = (
                        np.sum(weights.shape) / np.prod(weights.shape)
                    ) ** self.erk_power_scale
                    raw_probabilities[layer_idx] = raw_prob
                    divisor += raw_probabilities[layer_idx] * n_params
            eps = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_eps = max_prob * eps
            if max_prob_eps > 1:
                is_eps_valid = False
                for layer_idx, raw_prob in raw_probabilities.items():
                    if raw_prob == max_prob:
                        self._logger.info(
                            f"Sparsity of layer at index {layer_idx} set to 0.0"
                        )
                        dense_layers.add(layer_idx)
                        break
            else:
                is_eps_valid = True

        self._cache[sparsity] = []
        for layer_idx, layer_config in enumerate(groups):
            if layer_idx in dense_layers:
                layer_sparsity = 0.0
            else:
                layer_sparsity = 1 - (eps * raw_probabilities[layer_idx])
            layer_config["sparsity"] = layer_sparsity
            self._cache[sparsity].append(layer_sparsity)
        return groups
