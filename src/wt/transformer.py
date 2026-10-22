"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import MlpExtractor

GOAL_NET_OUT = []

import torch
import os
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

from wt import goal_net

DEPRECATED = True


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name)
    return (
        value is not None
        and value.strip().lower() not in ("", "0", "false", "no", "off")
    )


def _config_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.0
    resid_pdrop = float(os.environ.get('DROPOUT', 0.15))
    attn_pdrop = float(os.environ.get('DROPOUT', 0.15))

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past=None,
    ):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        allowed = self.mask[:, :, :T, :T].bool()
        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(B, T)}, got {tuple(attention_mask.shape)}"
                )
            valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)
            # Mask keys so a valid token can never attend to padding. Padding
            # queries receive a diagonal fallback solely to keep softmax rows
            # finite; their outputs are zeroed below.
            allowed = allowed & valid_tokens[:, None, None, :]
            diagonal = torch.eye(T, device=x.device, dtype=torch.bool)[None, None]
            padded_queries = ~valid_tokens[:, None, :, None]
            allowed = allowed | (padded_queries & diagonal)
        att = att.masked_fill(~allowed, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if attention_mask is not None:
            y = y * valid_tokens.unsqueeze(-1).to(y.dtype)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_actions = config.model_actions

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # modified decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(nn.Linear(config.n_embd, config.vocab_size), nn.Tanh())

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.obs_emb = nn.Sequential(nn.Linear(config.feature_dim, config.n_embd), nn.Tanh())
        if self.model_actions:
            self.act_emb = nn.Sequential(nn.Linear(config.act_dim, config.n_embd), nn.Tanh())
        # self.goal_emb = nn.Sequential(nn.Linear(2, config.n_embd))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, inputs):
        # ``inputs`` is kept as one object so GPT remains usable inside the
        # legacy nn.Sequential extractor.
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 2:
                obs, actions = inputs
                attention_mask = None
            elif len(inputs) == 3:
                obs, actions, attention_mask = inputs
            else:
                raise ValueError(
                    "GPT expects (observations, actions[, attention_mask])"
                )
        else:
            obs, actions, attention_mask = inputs, None, None
        if os.environ.get('debug'):
            print(obs[0, -1, :2])
        B, T, D = obs.shape
        assert D == self.config.feature_dim
        if attention_mask is None:
            attention_mask = torch.ones(
                (B, T), device=obs.device, dtype=torch.bool
            )
        else:
            attention_mask = attention_mask.to(device=obs.device)
            if attention_mask.ndim == 3 and attention_mask.shape[-1] == 1:
                attention_mask = attention_mask.squeeze(-1)
            if attention_mask.shape != (B, T):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(B, T)}, got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.bool()
        if actions is not None:
            actions = actions.view(B, T, -1)
            if self.model_actions and actions.shape[-1] != self.config.act_dim:
                raise ValueError(
                    f"Expected action_dim={self.config.act_dim}, "
                    f"got {actions.shape[-1]}"
                )
        if self.model_actions:
            token_embeddings = torch.zeros((B, 2 * T, self.config.n_embd), device = obs.device)
            token_embeddings[:, 0::2] = self.obs_emb(obs)
            token_embeddings[:, 1::2] = self.act_emb(actions) if actions is not None else 0
            assert token_embeddings.shape == (B, 2 * T, self.config.n_embd)
            token_attention_mask = attention_mask.repeat_interleave(2, dim=1)
        else:
            token_embeddings = self.obs_emb(obs) # (batch * block_size, n_embd)
            assert token_embeddings.shape == (B, T, self.config.n_embd)
            token_attention_mask = attention_mask

        if token_embeddings.shape[1] > self.block_size:
            raise ValueError(
                "Input sequence exceeds the configured Transformer block size: "
                f"{token_embeddings.shape[1]} > {self.block_size}"
            )
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = x * token_attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask=token_attention_mask)
        x = self.ln_f(x)
        x = x * token_attention_mask.unsqueeze(-1).to(x.dtype)
        logits = self.head(x)
        logits = logits * token_attention_mask.unsqueeze(-1).to(logits.dtype)
        if self.model_actions:
            assert logits.shape == (B, 2 * T, self.config.vocab_size)
            act = logits[:, 1::2]
            logits = logits[:, 0::2]
        else:
            act = None
            assert logits.shape == (B, T, self.config.vocab_size)

        logits = logits.view(B * T, self.config.vocab_size)

        return (logits, act) if not DEPRECATED else logits

class TransformerExtractor(MlpExtractor):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
        config=None,
        goal_network_ckpt=None,
        goal_network_config=None,
        goal_network_restore_mode=None,
        action_dim=None,
    ):
        config = {} if config is None else dict(config)
        super(TransformerExtractor, self).__init__(config.get('vocab_size', 512), net_arch, activation_fn, device)
        self.config = config

        # preprocessing data points w/ goal network
        self.goal_network = None
        self.cfg = self._normalize_goal_network_config(goal_network_config)
        self.goal_network_ckpt = goal_network_ckpt
        if goal_network_restore_mode not in (None, "embedded"):
            raise ValueError(
                "goal_network_restore_mode must be None or 'embedded'."
            )
        self.goal_network_restore_mode = goal_network_restore_mode
        if goal_network_ckpt is not None:
            if self.cfg is None:
                raise ValueError(
                    "goal_network_config is required when a waypoint network "
                    "is enabled; checkpoint filenames are not used to infer "
                    "state or goal dimensions"
                )
            if self.cfg["network_type"] == "manual":
                manual_config = {
                    key: value for key, value in self.cfg.items()
                    if key not in ("network_type", "waypoint_dim")
                }
                self.goal_network = goal_net.ManualGoalNetwork(**manual_config)
            elif goal_network_restore_mode == "embedded":
                self.goal_network = self._build_learned_goal_network()
            else:
                self.goal_network = self._load_learned_goal_network()
            self.goal_network.eval()
            for parameter in self.goal_network.parameters():
                parameter.requires_grad = False

        # if config is not specified, reverts to MLP only
        if config is not None:
            model_actions = _config_flag(
                config.get(
                    "model_actions",
                    _environment_flag("MODEL_ACTIONS"),
                )
            )
            configured_action_dim = config.get(
                'action_dim', config.get('act_dim', action_dim)
            )
            if configured_action_dim is None:
                if model_actions:
                    raise ValueError(
                        "action_dim must be supplied by the action space or "
                        "config when MODEL_ACTIONS is enabled"
                    )
                configured_action_dim = 0
            else:
                configured_action_dim = int(configured_action_dim)
            if action_dim is not None and configured_action_dim != int(action_dim):
                raise ValueError(
                    "Transformer action_dim conflicts with the environment "
                    f"action space: {configured_action_dim} != {int(action_dim)}"
                )
            context_length = int(config.get('context_length', 20))
            if context_length <= 0:
                raise ValueError("Transformer context_length must be positive")
            config = GPTConfig(vocab_size = config.get('vocab_size', 512), 
                               block_size = context_length * (1 + model_actions),
                               n_layer = int(config.get(
                                   "n_layer",
                                   os.environ.get("NUM_LAYERS", 2),
                               )),
                               n_head = config.get('n_head', 16), 
                               n_embd = config.get('n_embd', 128),
                               resid_pdrop=config.get(
                                   "resid_pdrop",
                                   float(os.environ.get("DROPOUT", 0.15)),
                               ),
                               attn_pdrop=config.get(
                                   "attn_pdrop",
                                   float(os.environ.get("DROPOUT", 0.15)),
                               ),
                               feature_dim = feature_dim + (
                                   self.cfg['waypoint_dim']
                                   if self.goal_network_ckpt else 0
                               ),
                               model_actions = model_actions,
                               act_dim = configured_action_dim)
            transformer_base = GPT(config).float()
        else:
            transformer_base = lambda x: x
        mlp_head = self.shared_net
        if DEPRECATED:
            self.shared_net = nn.Sequential(transformer_base, mlp_head)
        else:
            self.transformer_base, self.mlp_head = transformer_base, mlp_head

    @staticmethod
    def _normalize_goal_network_config(config):
        if config is None:
            return None
        config = dict(config)
        state_dim = config.pop("state_dim", config.get("obs_dim"))
        goal_dim = config.get("goal_dim")
        waypoint_dim = config.pop("waypoint_dim", goal_dim)
        network_type = config.get("network_type", "learned")
        if state_dim is None or goal_dim is None or waypoint_dim is None:
            raise ValueError(
                "goal_network_config must define state_dim/obs_dim, "
                "goal_dim, and waypoint_dim"
            )
        if network_type == "learned" and (
            config.get("hidden_dim") is None or config.get("max_T") is None
        ):
            raise ValueError(
                "A learned goal_network_config must define hidden_dim and max_T."
            )
        if int(waypoint_dim) != int(goal_dim):
            raise ValueError(
                "The official KForwardGoalNetwork emits goal_dim values, so "
                "waypoint_dim must equal goal_dim"
            )
        config["obs_dim"] = int(state_dim)
        config["goal_dim"] = int(goal_dim)
        config["waypoint_dim"] = int(waypoint_dim)
        config["network_type"] = network_type
        return config

    def _build_learned_goal_network(self):
        """Construct waypoint topology for an embedded outer state dict."""
        network_config = {
            key: value for key, value in self.cfg.items()
            if key not in ("network_type", "waypoint_dim")
        }
        return goal_net.KForwardGoalNetwork(**network_config)

    def _load_learned_goal_network(self):
        """Build and register a frozen waypoint network before policy loading."""
        try:
            checkpoint = torch.load(
                self.goal_network_ckpt,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.goal_network_ckpt,
                map_location="cpu",
            )
        checkpoint_hparams = checkpoint.get("hyper_parameters", {})
        checkpoint_hparams = (
            dict(checkpoint_hparams)
            if checkpoint_hparams is not None else {}
        )
        expected_metadata = {
            "obs_dim": self.cfg["obs_dim"],
            "goal_dim": self.cfg["goal_dim"],
            "hidden_dim": self.cfg["hidden_dim"],
            "max_T": self.cfg["max_T"],
            "reward": self.cfg.get("reward", False),
        }
        if self.cfg.get("reward", False):
            reward_metadata = {
                "reward_representation": self.cfg.get(
                    "reward_representation"
                ),
                "reward_normalization_const": self.cfg.get(
                    "reward_normalization_const"
                ),
                "reward_episode_horizon": self.cfg.get(
                    "reward_episode_horizon"
                ),
            }
            missing_config = [
                name for name, value in reward_metadata.items()
                if value is None
            ]
            if missing_config:
                raise ValueError(
                    "Reward waypoint config is missing semantic metadata: "
                    + ", ".join(missing_config)
                )
            expected_metadata.update(reward_metadata)

        def metadata_matches(name, actual, expected):
            return (
                np.isclose(actual, expected)
                if name == "reward_normalization_const"
                else actual == expected
            )

        # Never overwrite metadata already present in a checkpoint. Validate
        # present fields first, then supply only fields absent from old files.
        for name, expected in expected_metadata.items():
            if name in checkpoint_hparams and not metadata_matches(
                name, checkpoint_hparams[name], expected
            ):
                raise ValueError(
                    "Waypoint checkpoint metadata conflicts with the "
                    f"current dataset/config: {name}="
                    f"{checkpoint_hparams[name]!r}, expected {expected!r}"
                )
        load_config = {
            name: expected for name, expected in expected_metadata.items()
            if name not in checkpoint_hparams
        }
        network = goal_net.KForwardGoalNetwork.load_from_checkpoint(
            self.goal_network_ckpt,
            map_location="cpu",
            **load_config,
        )
        for name, expected in expected_metadata.items():
            actual = getattr(network, name)
            if not metadata_matches(name, actual, expected):
                raise ValueError(
                    "Waypoint checkpoint metadata conflicts with the "
                    f"current dataset/config: {name}={actual!r}, "
                    f"expected {expected!r}"
                )
        return network

    def forward(self, features):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = self.goal_extract(features)
        if not DEPRECATED:
            base_features, self.action_features = self.transformer_base(features)
            shared_latent = self.mlp_head(base_features)
        else:
            shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def goal_extract(self, features):
        if self.goal_network is None:
            return features
        if not isinstance(features, (tuple, list)):
            obs, actions, attention_mask = features, None, None
        elif len(features) == 2:
            obs, actions = features
            attention_mask = None
        elif len(features) == 3:
            obs, actions, attention_mask = features
        else:
            raise ValueError(
                "Transformer features must be "
                "(observations, actions[, attention_mask])"
            )
        expected_input_dim = self.cfg["obs_dim"] + self.cfg["goal_dim"]
        if obs.shape[-1] != expected_input_dim:
            raise ValueError(
                "Waypoint network input dimension does not match "
                f"goal_network_config: {obs.shape[-1]} != "
                f"{self.cfg['obs_dim']} + {self.cfg['goal_dim']}"
            )
        with torch.no_grad():
            new_goals = self.goal_network(obs).detach()
            if new_goals.shape[-1] != self.cfg["waypoint_dim"]:
                raise ValueError(
                    "Waypoint network output dimension does not match "
                    f"waypoint_dim={self.cfg['waypoint_dim']}: "
                    f"got {new_goals.shape[-1]}"
                )
            # print(new_goals[0, -1], obs[0, -1, :2])
            if os.environ.get('TRACK_GOALS'):
                global GOAL_NET_OUT
                GOAL_NET_OUT.extend(
                    new_goals[:, -1].cpu().reshape(
                        (-1, self.cfg["waypoint_dim"])
                    ).tolist()
                )
        return torch.cat([obs, new_goals], dim = -1), actions, attention_mask
