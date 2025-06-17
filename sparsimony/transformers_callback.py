import logging
from typing import Any, Dict, List
import os
import contextlib

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

try:
    import wandb

    _WANDB_INSTALLED = True
except ImportError:
    wandb = None
    _WANDB_INSTALLED = False

from transformers.trainer_callback import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from sparsimony.dst.base import DSTMixin

logger = logging.getLogger(__name__)


class SparsimonyCallback(TrainerCallback):
    SPARSIFIER_STATE_FILE = "sparsifier_state.pt"
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        sparsifier: DSTMixin,
        squash_mask_on_train_end: bool = True,
        # json_serializable: bool = False,
    ):
        self.sparsifier = sparsifier
        self.squash_mask_on_train_end = squash_mask_on_train_end

    @property
    def step_count(self) -> int:
        return self.sparsifier._step_count

    @step_count.setter
    def step_count(self, value: int):
        self.step_count = value
        self.sparsifier._step_count = value
        logger.info(
            f"Step count set to {value} in SparsimonyCallback. "
            "You should only see this message when loading a checkpoint."
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._logger.info(
            "Sparsimony sparsifier settings:\n%s", self.sparsifier
        )
        return None

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.sparsifier.step():
            self._logger.info(self.sparsifier)
        return None  # We do not modify control object

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.squash_mask_on_train_end:
            self.sparsifier.squash_mask()
            self.logger.info("Sparsifier mask squashed on train end.")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = args.output_dir

        # The name of the checkpoint folder is "checkpoint-STEP"
        # The constant PREFIX_CHECKPOINT_DIR is simply "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_dir = os.path.join(output_dir, checkpoint_folder)

        state_file = os.path.join(checkpoint_dir, self.SPARSIFIER_STATE_FILE)
        state_dict = self.sparsifier.state_dict()
        with open(state_file, "wb") as f:
            torch.save(state_dict, f)
        self._logger.info(f"Saved sparsifier state to: {state_file}")
        return None

    def load_checkpoint(self, resume_from_checkpoint: bool | str) -> None:
        """
        Load the sparsifier state from a checkpoint directory.
        """
        if isinstance(resume_from_checkpoint, str):
            checkpoint_dir = resume_from_checkpoint
        else:
            checkpoint_dir = get_last_checkpoint(resume_from_checkpoint)
        with open(
            os.path.join(
                checkpoint_dir, SparsimonyCallback.SPARSIFIER_STATE_FILE
            ),
            "rb",
        ) as f:
            state_dict = torch.load(f, weights_only=False)
        self.sparsifier.load_state_dict(state_dict)
        logger.info(
            f"Loaded sparsifier state from checkpoint at {checkpoint_dir}"
        )


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
        if torch.distributed.is_initialized():
            rank = os.environ["RANK"]
            if rank != 0:
                return None
        if _WANDB_INSTALLED:
            wandb.log({"wikitext_ppl": ppl["mean_perplexity"]})
        else:
            self._logger.info(
                f"Wikitext2 PPL: {ppl['mean_perplexity']:.4f} "
                f"({len(ppl['perplexities'])} sequences)"
            )
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
        # if batch_size > 1 (which generally leads to padding being required)
        # and if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token
            assert len(existing_special_tokens) > 0, (
                "If batch_size > 1, model must have at least one special token "
                "to use for padding. Please use a different model or set "
                "batch_size=1."
            )
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens(
                {"pad_token": existing_special_tokens[0]}
            )

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert tokenizer.bos_token is not None, (
                "Input model must already have a BOS token if using "
                "add_start_token=True. Please use a different model, or set "
                "add_start_token=False"
            )
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


@contextlib.contextmanager
def temporarily_disable_deepspeed_zero3(training_arguments):
    if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
        unset_hf_deepspeed_config()
        yield
        set_hf_deepspeed_config(training_arguments.hf_deepspeed_config)
    else:
        yield
