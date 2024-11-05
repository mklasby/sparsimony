import logging
from collections.abc import Iterable
from typing import Any, Dict, Callable, List, Tuple, Type
import copy
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from transformers.trainer_callback import (
    TrainerCallback,
    ExportableState,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from accelerate.optimizer import AcceleratedOptimizer
import wandb

from transformers.trainer_callback import (
    TrainerCallback,
    ExportableState,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)


# class SparsimonyCallback(TrainerCallback, ExportableState):
class SparsimonyCallback(TrainerCallback):
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        sparsifier,
        # sparsifier_class: type,
        # sparsifier_config: List[Dict[str, Any]],
        # sparsifier_kwargs: Dict[str, Any] | None = None,
        # json_serializable: bool = False,
    ):
        # If None, we delay initalization until we have scope on Trainer attributes
        # self.sparsifier_class = sparsifier_class
        # self.sparsifier_config = sparsifier_config
        # if sparsifier_kwargs is None:
        #     sparsifier_kwargs = {}
        # self.sparsifier_kwargs = sparsifier_kwargs
        self.sparsifier = sparsifier
        # self.sparse_optim = sparse_optim
        # # following are required to load ckpt, will be initalized by self.state()
        # self.sparse_optim_state: dict | None = None
        # self.sparse_optim_type: type | None = None
        # self.sparse_optim_kwargs: dict | None = None
        # self.json_serializable = json_serializable

    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, optimizer, **kwargs):
    #     self.sparsifier = self.sparsifier_class(optimizer=optimizer, **self.sparsifier_kwargs)
    #     self.sparsifier.prepare(model, self.sparsifier_config)
    #     return None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, optimizer, **kwargs):
        print(self.sparsifier)
        return None

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.sparsifier.step():
            print(self.sparsifier)
        return None  # We do not modify control object

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.sparsifier.squash_masks()

    # def state(self) -> dict:
    #     state_dict = self.sparsifier.state_dict()
    #     state = dict(
    #         args={},
    #         attributes=dict(
    #             state_dict=state_dict
    #         ),
    #     )
    #     return state

    # @classmethod
    # def from_state(cls, state):
    #     self.sparsifier.load_state_dict(**state)
    #     instance = cls(**state["args"])
    #     for k, v in state["attributes"].items():
    #         setattr(instance, k, v)
    #     instance.sparse_optim_type = cls._SPARSE_OPTIM_TYPE_MAP[
    #         instance.sparse_optim_type
    #     ]
    #     if instance.json_serializable:
    #         instance.sparse_optim_state = cls._unserialize_state_dict(
    #             instance.sparse_optim_state
    #         )
    #     return instance
        

    # def state(self) -> dict:
    #     if self.sparse_optim is None:
    #         raise AttributeError(self._raise_msg_attrs_not_none())
    #     if not self.json_serializable:
    #         # Sync with transformers.trainer_callback.TrainerState
    #         self.sparse_optim_state = self.sparse_optim.state_dict()
    #     else:
    #         self.sparse_optim_state = self._get_serializable_sparse_optim_state()
    #     self.sparse_optim_type = str(type(self.sparse_optim))
    #     self.sparse_optim_kwargs = self._get_sparse_optim_kwargs()
    #     state = dict(
    #         args={},
    #         attributes=dict(
    #             sparse_optim_state=self.sparse_optim_state,
    #             sparse_optim_type=self.sparse_optim_type,
    #             sparse_optim_kwargs=self.sparse_optim_kwargs,
    #         ),
    #     )
    #     return state

    # @classmethod
    # def from_state(cls, state):
    #     instance = cls(**state["args"])
    #     for k, v in state["attributes"].items():
    #         setattr(instance, k, v)
    #     instance.sparse_optim_type = cls._SPARSE_OPTIM_TYPE_MAP[
    #         instance.sparse_optim_type
    #     ]
    #     if instance.json_serializable:
    #         instance.sparse_optim_state = cls._unserialize_state_dict(
    #             instance.sparse_optim_state
    #         )
    #     return instance

    # def _init_sparse_optim_from_kwargs(self, model, optimizer, **kwargs) -> None:
    #     if self.sparse_optim_kwargs is None or self.sparse_optim_type is None:
    #         raise AttributeError(self._raise_msg_attrs_not_none())
    #     if isinstance(optimizer, AcceleratedOptimizer):
    #         optimizer = optimizer.optimizer
    #     optimizer_states_to_track = optimizer_trackers(optimizer)
    #     self.sparse_optim = self.sparse_optim_type(
    #         params=model.paramters(),
    #         optimizers=(optimizer, optimizer_states_to_track),
    #         **self.sparse_optim_kwargs,
    #     )
    #     self.sparse_optim.load_state_dict(self.sparse_optim_state)

    # def _get_sparse_optim_kwargs(self) -> dict:
    #     if self.sparse_optim is None:
    #         raise AttributeError(self._raise_msg_attrs_not_none())
    #     return self.sparse_optim.defaults

    # def _raise_msg_attrs_not_none(self) -> str:
    #     msg = ""
    #     if self.sparse_optim_type is None:
    #         msg += "sparse_optim_type is None when a type was expected! "
    #     if self.sparse_optim_kwargs is None:
    #         msg += "sparse_optim_kwargs is None when a dict was expected! "
    #     if self.sparse_optim is None:
    #         msg += "sparse_optim is None when a cbsparse.optimizer was expected!"
    #     return msg

    # def _get_serializable_sparse_optim_state(self):
    #     self._logger.info("Serializing sparse_optim state_dict...")
    #     start = time.time()
    #     if self.sparse_optim is None:
    #         raise AttributeError(self._raise_msg_attrs_not_none())
    #     state_dict = self.sparse_optim.state_dict()
    #     state_dict = self._serializable_state_dict(state_dict)
    #     end = time.time()
    #     self._logger.info(
    #         f"Sparse optim state_dict serialized in {end - start} seconds"
    #     )
    #     return state_dict

    # def _serializable_state_dict(self, state_dict: dict) -> dict:
    #     state_dict = copy.deepcopy(state_dict)
    #     for k, v in state_dict.items():
    #         if isinstance(v, torch.Tensor):
    #             state_dict[k] = self._serialize_tensor(v)
    #         elif isinstance(v, Iterable):
    #             state_dict[k] = self._serialize_iterable(v)
    #     return state_dict

    # def _serialize_iterable(self, iterable: Iterable) -> Iterable:
    #     iterable_serializers: Dict[Type, Callable] = {
    #         list: self._serialize_list,
    #         dict: self._serialize_dict,
    #         str: lambda x: x,
    #     }
    #     return iterable_serializers[type(iterable)](iterable)

    # def _serialize_dict(self, d: dict) -> dict:
    #     for k, v in d.items():
    #         if isinstance(v, torch.Tensor):
    #             d[k] = self._serialize_tensor(v)
    #         elif isinstance(v, Iterable):
    #             d[k] = self._serialize_iterable(v)
    #     return d

    # def _serialize_list(self, l: list) -> list:
    #     serialized_list = []
    #     for el in l:
    #         if isinstance(el, torch.Tensor):
    #             serialized_list.append(self._serialize_tensor(el))
    #         elif isinstance(el, Iterable):
    #             serialized_list.append(self._serialize_iterable(el))
    #         else:
    #             serialized_list.append(el)
    #     return serialized_list

    # def _serialize_tensor(self, t: torch.Tensor) -> list:
    #     return t.cpu().detach().numpy().tolist()

    # @classmethod
    # def _unserialize_state_dict(cls, sparse_optim_state: dict) -> dict:
    #     for state_idx in sparse_optim_state["state"]:
    #         for k, v in sparse_optim_state["state"][state_idx].items():
    #             if isinstance(v, list):
    #                 sparse_optim_state["state"][state_idx][k] = torch.Tensor(v)
    #     return sparse_optim_state


class PplCallBack(TrainerCallback):
    _logger = logging.getLogger(__name__)

    def __init__(self, dataset: List[str]):
        self.dataset = dataset
        self.packed_sequences = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs["model"]
        tokenizer = kwargs["processing_class"]
        ppl = self.compute_ppl(model, tokenizer)
        wandb.log({"wikitext_ppl": ppl["mean_perplexity"]})
        return None

    def compute_ppl(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
        add_start_token: bool = True,
        max_length=2048,
    ) -> Dict[str, Any]:
        if self.packed_sequences is None:
            self.packed_sequences = self._encode_dataset(
                tokenizer, batch_size, add_start_token, max_length
            )
        device = model.device
        packed_sequences = self.packed_sequences.to(device)
        ppls = []
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        for batch in tqdm(packed_sequences, desc="Wikitext2 PPL..."):
            if len(batch.shape) < 2:
                batch = batch.unsqueeze(dim=0)
            attn_mask = torch.ones_like(batch)
            labels = batch

            with torch.no_grad():
                # import pdb
                # breakpoint()
                out_logits = model(batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def _encode_dataset(
        self,
        tokenizer,
        batch_size: int = 8,
        add_start_token: bool = True,
        max_length=2048,
    ) -> torch.Tensor:
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            "\n".join(self.dataset["text"]),  # sparsegpt version
            add_special_tokens=False,  # no bos, we manually add
            return_tensors="pt",
            return_attention_mask=False,
        )
        encoded_texts = encodings["input_ids"][0]

        n_samples = len(encoded_texts) // max_tokenized_len
        packed_sequences = torch.zeros(n_samples, max_length, dtype=torch.long)
        for i in range(n_samples):
            cursor = 0
            if add_start_token:
                packed_sequences[i, cursor] = tokenizer.bos_token_id
                cursor += 1
            packed_sequences[i, cursor : max_tokenized_len + 1] = encoded_texts[
                i * max_tokenized_len : (i + 1) * max_tokenized_len
            ]
        return packed_sequences
