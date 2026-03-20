"""C-RADIOv4 encoder wrapper for RAE Stage 1 training.

C-RADIOv4 (e.g., C-RADIOv4-SO400M) is a ViT-SO400M distilled from three
teachers (siglip2-g, dino_v3_7b, sam3).  Its RADIOModel.forward() expects
[0,1] float images and applies OpenAI-CLIP normalisation internally via the
InputConditioner.

RAE's encode() pipeline:
  [0,1] image
    -> bicubic resize to encoder_input_size
    -> subtract encoder_mean / divide encoder_std   (from AutoImageProcessor)
    -> encoder.forward(x)                           <- this wrapper is called here
    -> (B, N, C) patch tokens

The AutoImageProcessor for C-RADIOv4 is a CLIPImageProcessor whose
image_mean / image_std are the OpenAI CLIP constants.  So by the time
forward() is called, x has been CLIP-normalised by RAE.

This wrapper therefore *inverts* that normalisation (recovering [0,1]) before
passing the image to RADIOModel, which applies its own InputConditioner
(= CLIP normalisation) internally.  This ensures the encoder always sees
correctly normalised input regardless of what RAE does upstream.
"""

import sys
import os
from pathlib import Path

import torch
from torch import nn

from . import register_encoder

# ---------------------------------------------------------------------------
# Locate the RADIO source tree and add it to sys.path so we can import it
# without installing the package.  We look next to the RAE project root.
# ---------------------------------------------------------------------------
_RAE_SRC = Path(__file__).resolve().parents[3]          # …/RAE/src/  -> …/RAE/
_RADIO_ROOT = _RAE_SRC.parent / "RADIO"                  # …/vision_encoders/RADIO

if str(_RADIO_ROOT) not in sys.path:
    sys.path.insert(0, str(_RADIO_ROOT))


@register_encoder()
class CRadiov4withNorm(nn.Module):
    """Wrapper around C-RADIOv4 (HuggingFace RADIOModel) for RAE Stage 1.

    Parameters
    ----------
    radio_path : str
        Path to the local HuggingFace snapshot of C-RADIOv4
        (i.e. the directory containing ``config.json`` and
        ``model.safetensors``).
    input_size : int
        The spatial resolution fed to RADIO.  Must be a multiple of
        ``patch_size * window_size`` (= 16 for C-RADIOv4-SO400M).
        Defaults to 512 (the model's preferred resolution).
    """

    def __init__(
        self,
        radio_path: str,
        input_size: int = 512,
    ):
        super().__init__()

        # Import here so the RADIO path addition above takes effect first.
        from radio.hf_model import RADIOModel  # type: ignore

        self.radio_model = RADIOModel.from_pretrained(
            radio_path, local_files_only=True
        )
        self.radio_model.requires_grad_(False)

        # Expose the attributes required by Stage1Protocal / RAE.__init__
        self.patch_size: int = self.radio_model.patch_size          # 16
        self.hidden_size: int = self.radio_model.radio_model.embed_dim  # 1152

        self._input_size = input_size

        # Cache the InputConditioner's mean/std so we can invert RAE's
        # pre-normalisation back to [0,1].  The vectors have shape (3,1,1).
        ic = self.radio_model.input_conditioner
        # norm_mean / norm_std were stored after dividing by input_scale (1.0),
        # so they ARE the CLIP mean/std.
        self.register_buffer("_clip_mean", ic.norm_mean.float())  # (3,1,1)
        self.register_buffer("_clip_std",  ic.norm_std.float())   # (3,1,1)

    # ------------------------------------------------------------------
    # RAE calls  self.encoder(x)  where x has already been normalised by
    # the processor stats loaded from encoder_config_path.
    # For C-RADIOv4 the processor has do_normalize=False but its image_mean
    # / image_std are set to the OpenAI CLIP constants, so RAE applies CLIP
    # normalisation before calling us.  RADIO's InputConditioner then
    # applies CLIP normalisation *again*, which would be wrong.
    #
    # To avoid double-normalisation we invert RAE's pre-normalisation here
    # (recovering [0,1] input) and let RADIOModel handle the rest.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 3, H, W) float tensor.
            Values in CLIP-normalised space (output of RAE's encode() path).

        Returns
        -------
        (B, N, C) spatial patch tokens, CLS/summary tokens stripped.
        """
        # Invert RAE's CLIP normalisation to recover [0,1] pixel values.
        x = x * self._clip_std.to(x) + self._clip_mean.to(x)
        x = x.clamp(0.0, 1.0)

        # RADIOModel.forward() raises if resolution is not a multiple of
        # min_resolution_step.  RAE already resizes to encoder_input_size
        # which should be a multiple of patch_size (16), so this is met.
        summary, features = self.radio_model(x)   # features: (B, N, C)
        return features