"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
import logging
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipIntermediateOutput,
    BlipOutputWithLogits,
)
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from lavis.models.embeddings import tSP_Embedding, Cosine_Embedding


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@registry.register_model("blip_classification")
class BlipClassification(BlipBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {        
        "Vit-base-tSPEmbedding-ce-imagenetlt": "configs/models/blip/blip_classification_base_tSPEmbedding_ce_imagenetlt.yaml",
        "Vit-large-tSPEmbedding-ce-imagenetlt": "configs/models/blip/blip_classification_large_tSPEmbedding_ce_imagenetlt.yaml",

        "Vit-base-tSPEmbedding-ce-inat18":  "configs/models/blip/blip_classification_base_tSPEmbedding_ce_inat18.yaml",
        "Vit-large-tSPEmbedding-ce-inat18": "configs/models/blip/blip_classification_large_tSPEmbedding_ce_inat18.yaml",

        "Vit-base-tSPEmbedding-ce-places": "configs/models/blip/blip_classification_base_tSPEmbedding_ce_places.yaml",
        "Vit-large-tSPEmbedding-ce-places":"configs/models/blip/blip_classification_large_tSPEmbedding_ce_places.yaml"
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        cls_head,
        criterion,
        num_classes,
        freeze_vit,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=40,
        use_distill=True,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.use_distill = use_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder
        self.cls_head = cls_head
        self.criterion = criterion

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        if self.use_distill:
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            self.text_encoder_m = deepcopy(self.text_encoder)
            self.cls_head_m = deepcopy(self.cls_head)

            self.momentum = momentum
            self.alpha = alpha

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.cls_head, self.cls_head_m],
            ]

            self.copy_params()

        self.max_txt_len = max_txt_len

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        sentences = samples["text_input"]
        sentences = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        samples.update({"tokenized_text": sentences})

        targets = samples["label"]
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            samples["tokenized_text"], image_embeds
        )

        prediction = self.cls_head[:2](encoder_output.last_hidden_state[:, 0, :])

        if is_train:
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(samples["image"])
                    encoder_output_m = self.text_encoder_m.forward_automask(
                        samples["tokenized_text"], image_embeds_m
                    )

                    prediction_m = self.cls_head_m(
                        encoder_output_m.last_hidden_state[:, 0, :]
                    )

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"],
                    iters=samples["iters"],
                    num_iters_per_epoch=samples["num_iters_per_epoch"],
                )

                loss = (1 - alpha) * self.criterion(
                    prediction, targets
                ) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1),
                    dim=1,
                ).mean()
            else:
                loss = self.criterion(prediction, targets)

            # return {"loss": loss}
            return BlipOutputWithLogits(
                loss=loss,
                intermediate_output=BlipIntermediateOutput(
                    image_embeds=image_embeds,
                    image_embeds_m=image_embeds_m,
                    encoder_output=encoder_output,
                    # encoder_output_m=encoder_output_m,
                ),
                logits=prediction,
                logits_m=prediction_m,
            )

        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        if hasattr(self.criterion,'predict'):
            output['predictions'] = self.criterion.predict(output['predictions'])
        return output

    @classmethod
    def from_config(cls, cfg=None):
        visionencoder = cfg.get("vision_encoder", "vit")
        if 'vit' in visionencoder.lower():
            image_encoder = VisionTransformerEncoder.from_config(cfg)
        elif 'resnet' in visionencoder.lower():
            image_encoder = ResNetEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)

        use_distill = cfg.get("use_distill", True)
        momentum = cfg.get("momentum", 0.995)
        num_classes = cfg.get("num_classes", -1)
        alpha = cfg.get("alpha", 0.4)
        max_txt_len = cfg.get("max_txt_len", 512)
        cls_head = cfg.get("cls_head", "mlp")
        criterion = cfg.get("criterion", "ce")
        freeze_vit = cfg.get("freeze_vit", True)
        
        print("BLIP CLASSIFICATION")
        print("use_distill: ", use_distill)
        print("num_classes: ",num_classes)
        print("max_txt_len: ",max_txt_len)
        print("cls_head:  ", cls_head)
        print("criterion: ", criterion)
        print("freeze_vit: ", freeze_vit)

        hidden_size = text_encoder.config.hidden_size

        if cls_head == 'mlp':
            cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )
        elif cls_head == "tSP":
            print("Loading Cls head: tSP")
            cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                tSP_Embedding(hidden_size, num_classes),
            )
        elif cls_head == "Cosine":
            cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                Cosine_Embedding(hidden_size, num_classes),
            )


        else:
            cls_head = None
            assert "No cls_head!"

        if criterion == 'ce':
            print("using ce criterion")
            criterion = nn.CrossEntropyLoss( ignore_index=-100, reduction='mean')
        else:
            criterion = None
            assert "No criterion!"




        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            cls_head = cls_head,
            freeze_vit = freeze_vit,
            criterion=criterion,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        model.load_checkpoint_from_config(cfg)

        # pretrain_path = cfg.get("pretrained", None)
        # if pretrain_path is not None:
        #     msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        return model


