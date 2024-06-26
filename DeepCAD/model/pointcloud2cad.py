import sys

sys.path.append("..")
from model.autoencoder import CADTransformer, Decoder
from model.pointcloudEncoder import PointNet2

from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .layers.transformer import *
from .model_utils import (
    _get_group_mask,
    _get_key_padding_mask,
    _get_padding_mask,
    _make_batch_first,
    _make_seq_first,
)


class PointCloud2CAD(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""

    def __init__(self, cfg):
        super().__init__()

        self.pc_enc = PointNet2()
        # self.bottleneck = trainer_ae.net.bottleneck
        # self.autoencoder = CADTransformer(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, points):
        z = self.pc_enc(points)
        z = z.unsqueeze(1)
        # z = self.bottleneck(z)

        z = _make_seq_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {"command_logits": out_logits[0], "args_logits": out_logits[1]}

        return res
