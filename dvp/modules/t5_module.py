import torch
import torch.nn as nn
import pytorch_lightning as pl
from dvp.modules import objectives, dvp_utils
import clip
from transformers import T5Model
import math
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Cross_Attention(nn.Module):
    def __init__(self, hidden_size, drop_rate, num_heads):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_rate)


    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        reduction_factor = 8
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        nn.init.normal_(self.down_sampler.weight, std=1e-2)
        nn.init.zeros_(self.down_sampler.bias)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        nn.init.normal_(self.up_sampler.weight, std=1e-2)
        nn.init.zeros_(self.up_sampler.bias)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        output = x + z
        return output

class DVP_T5(pl.LightningModule):
    def __init__(self, config, kab_val_dataset=None):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = 'T5'
        self.t5_model = T5Model.from_pretrained(config['tokenizer'])
        self.t5_encoder = self.t5_model.encoder
        self.t5_decoder = self.t5_model.decoder

        self.visumodel = clip.load(config["clip_model"], jit=False, device=torch.device("cpu"))[0].visual
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.search_stage = config['search_stage']
        self.use_adapter = config['use_adapter']

        if self.use_adapter:

            self.encoder_adapter = nn.ModuleList(
                [Adapter(config["hidden_size"]) for _ in range(2 * len(self.t5_encoder.block))]
            )

            self.decoder_adapter = nn.ModuleList(
                [Adapter(config["hidden_size"]) for _ in range(3 * len(self.t5_decoder.block))]
            )
            for name, param in self.t5_encoder.named_parameters():
                if 'layer_norm' not in name:
                    param.requires_grad = False

            for name, param in self.t5_decoder.named_parameters():
                if 'layer_norm' not in name:
                    param.requires_grad = False

        if self.search_stage:

            self.search_sample = config['search_sample']
            self.token_type_embeddings_options = nn.ModuleList([nn.Embedding(2, config["hidden_size"])
                                                                for _ in range(config["num_layers"])])

            self.cross_attn_options = nn.ModuleList([Cross_Attention(config["hidden_size"],
                                                                     config["drop_rate"],
                                                                     config["num_heads"])
                                                     for _ in range(config["num_layers"])])

            self.kab_val_dataset = kab_val_dataset
            self.kab_val_iter = None

            self.layer_alphas = nn.Parameter(torch.zeros(12))

            self.kab_update_state = False
            self.active_index_buffer = []
            self.up_weight_buffer = []
            self.log_probs_buffer = []
            self.arch_loss_buffer = []
            self.loss_buffer = []
            self.score_buffer = []
            self.reward_buffer = []
            self.kab_print_info = 0

        else:
            self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
            self.cross_attn = Cross_Attention(config["hidden_size"], config["drop_rate"], config["num_heads"])
            self.insert_layer = config['insert_layer']


        for param in self.visumodel.parameters():
            param.requires_grad = False

        # ===================== V&l tasks ===================== #

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["gqa"] > 0:
            vs = self.hparams.config["gqa_label_size"]
            self.gqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.gqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["snli_ve"] > 0:
            self.snli_ve_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_ve_classifier.apply(objectives.init_weights)


        dvp_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== Test ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)


    def infer(
        self,
        batch,
        image_token_type_idx=1,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        image = batch[imgkey].cuda(torch.cuda.current_device())

        text_input_ids = batch[f"text_input_ids"].cuda(torch.cuda.current_device())
        text_attention_mask = batch[f"text_attention_mask"].cuda(torch.cuda.current_device())

        x = self.visumodel.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visumodel.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                dtype=x.dtype, device=x.device), x],
                      dim=1)
        x = x + self.visumodel.positional_embedding.to(x.dtype)
        x = self.visumodel.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visumodel.transformer(x)
        x = x.permute(1, 0, 2)

        # T5 encoder for processing text features
        if not self.use_adapter:
            encoder_hidden_states = self.compute_encoder_output(text_input_ids,text_attention_mask)

        else:
            encoder_hidden_states = self.compute_adapter_encoder_output(text_input_ids,text_attention_mask)

        # T5 decoder input vector
        decoder_input_ids, decoder_hidden_states = self.get_decoder_input(text_input_ids)

        # Define attention mask
        mask_seq_length = decoder_hidden_states.shape[1]
        decoder_attention_mask = torch.ones(text_input_ids.shape[0], mask_seq_length, device=text_input_ids.device)
        attention_mask = self.t5_model.get_extended_attention_mask(decoder_attention_mask, decoder_attention_mask.shape)

        mask_seq_length = 1 + decoder_hidden_states.shape[1]
        decoder_attention_mask = torch.ones(text_input_ids.shape[0], mask_seq_length, device=text_input_ids.device)
        merge_attention_mask = self.t5_model.get_extended_attention_mask(decoder_attention_mask,
                                                                            decoder_attention_mask.shape)

        encoder_extended_attention_mask = self.t5_model.invert_attention_mask(text_attention_mask)

        past_key_values = [None] * len(self.t5_decoder.block)
        head_mask = None
        head_mask = self.t5_decoder.get_head_mask(head_mask, self.t5_decoder.config.num_layers)
        cross_attn_head_mask = None
        cross_attn_head_mask = self.t5_decoder.get_head_mask(cross_attn_head_mask, self.t5_decoder.config.num_layers)


        if self.training:
            if self.search_stage:
                if not self.kab_update_state:
                    insert_layer = random.randint(0, len(self.layer_alphas) - 1)

                else:
                    probs = F.softmax(self.layer_alphas, dim=0)
                    insert_layer = torch.multinomial(probs.data, 1)[0].item()
                    self.active_index_buffer.append(insert_layer)
                    up_weight = probs[insert_layer] * (1 - probs[insert_layer])
                    self.up_weight_buffer.append(up_weight)
                    log_probs = torch.log(probs[insert_layer])
                    self.log_probs_buffer.append(log_probs)
            else:
                insert_layer = self.insert_layer

            for i, (layer_module, past_key_value) in enumerate(zip(self.t5_decoder.block, past_key_values)):
                if i < insert_layer:
                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           attention_mask)

                elif i == insert_layer:
                    text_pooling_token = self.pooling(encoder_hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
                    if self.search_stage:
                        visual_prompt = self.cross_attn_options[i](x, x, text_pooling_token, None)
                        decoder_hidden_states, visual_prompt = (decoder_hidden_states + self.token_type_embeddings_options[i](
                            torch.zeros_like(decoder_input_ids, device=text_input_ids.device).long()),
                                                          visual_prompt + self.token_type_embeddings_options[i](
                                                              torch.full_like(torch.ones(visual_prompt.shape[0],
                                                                                         visual_prompt.shape[1]),
                                                                              image_token_type_idx).long().to(
                                                                  text_input_ids.device)))
                    else:
                        visual_prompt = self.cross_attn(x, x, text_pooling_token, None)
                        decoder_hidden_states, visual_prompt = (
                        decoder_hidden_states + self.token_type_embeddings(
                            torch.zeros_like(decoder_input_ids, device=text_input_ids.device).long()),
                        visual_prompt + self.token_type_embeddings(
                            torch.full_like(torch.ones(visual_prompt.shape[0],
                                                       visual_prompt.shape[1]),
                                            image_token_type_idx).long().to(
                                text_input_ids.device)))
                    decoder_hidden_states = torch.cat([visual_prompt, decoder_hidden_states], dim=1)

                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   merge_attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           merge_attention_mask)

                else:
                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   merge_attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           merge_attention_mask)


        else:
            if self.search_stage:
                probs = F.softmax(self.layer_alphas, dim=0).data.cpu().numpy()
                insert_layer = int(np.argmax(probs))
            else:
                insert_layer = self.insert_layer

            for i, (layer_module, past_key_value) in enumerate(zip(self.t5_decoder.block, past_key_values)):
                if i < insert_layer:
                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           attention_mask)

                elif i == insert_layer:
                    text_pooling_token = self.pooling(encoder_hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
                    if self.search_stage:
                        visual_prompt = self.cross_attn_options[i](x, x, text_pooling_token, None)
                        decoder_hidden_states, visual_prompt = (
                        decoder_hidden_states + self.token_type_embeddings_options[i](
                            torch.zeros_like(decoder_input_ids, device=text_input_ids.device).long()),
                        visual_prompt + self.token_type_embeddings_options[i](
                            torch.full_like(torch.ones(visual_prompt.shape[0],
                                                       visual_prompt.shape[1]),
                                            image_token_type_idx).long().to(
                                text_input_ids.device)))
                    else:
                        visual_prompt = self.cross_attn(x, x, text_pooling_token, None)
                        decoder_hidden_states, visual_prompt = (
                            decoder_hidden_states + self.token_type_embeddings(
                                torch.zeros_like(decoder_input_ids, device=text_input_ids.device).long()),
                            visual_prompt + self.token_type_embeddings(
                                torch.full_like(torch.ones(visual_prompt.shape[0],
                                                           visual_prompt.shape[1]),
                                                image_token_type_idx).long().to(
                                    text_input_ids.device)))
                    decoder_hidden_states = torch.cat([visual_prompt, decoder_hidden_states], dim=1)
                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   merge_attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           merge_attention_mask)

                else:
                    if not self.use_adapter:
                        decoder_hidden_states = self.compute_layer(layer_module,
                                                                   past_key_value,
                                                                   head_mask[i],
                                                                   cross_attn_head_mask[i],
                                                                   decoder_hidden_states,
                                                                   encoder_hidden_states,
                                                                   encoder_extended_attention_mask,
                                                                   merge_attention_mask)
                    else:
                        decoder_hidden_states = self.compute_adapter_layer(i,
                                                                           past_key_value,
                                                                           head_mask[i],
                                                                           cross_attn_head_mask[i],
                                                                           decoder_hidden_states,
                                                                           encoder_hidden_states,
                                                                           encoder_extended_attention_mask,
                                                                           merge_attention_mask)


        decoder_hidden_states = self.t5_decoder.final_layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.t5_decoder.dropout(decoder_hidden_states)

        last_hidden_state = decoder_hidden_states.view(decoder_hidden_states.shape[0], -1, self.t5_model.config.d_model)[:, -1]

        ret = {
            "decoder_feats": last_hidden_state,
        }

        return ret


    def compute_adapter_layer(self,i,past_key_value,
                      layer_head_mask,cross_attn_layer_head_mask,
                      decoder_hidden_states,encoder_hidden_states,
                      encoder_extended_attention_mask,attention_mask):
        normed_hidden_states = self.t5_decoder.block[i].layer[0].layer_norm(decoder_hidden_states)
        attention_output = self.t5_decoder.block[i].layer[0].SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=None,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=None,
            output_attentions=True,
        )
        y = attention_output[0]
        y = self.decoder_adapter[3 * i](y)
        decoder_hidden_states = decoder_hidden_states + self.t5_decoder.block[i].layer[0].dropout(y)

        normed_hidden_states = self.t5_decoder.block[i].layer[1].layer_norm(decoder_hidden_states)
        attention_output = self.t5_decoder.block[i].layer[1].EncDecAttention(
            normed_hidden_states,
            mask=encoder_extended_attention_mask,
            key_value_states=encoder_hidden_states,
            position_bias=None,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=None,
            query_length=None,
            output_attentions=True,
        )

        y = attention_output[0]
        y = self.decoder_adapter[3 * i + 1](y)
        decoder_hidden_states = decoder_hidden_states + self.t5_decoder.block[i].layer[1].dropout(y)

        forwarded_states = self.t5_decoder.block[i].layer[-1].layer_norm(decoder_hidden_states)
        forwarded_states = self.t5_decoder.block[i].layer[-1].DenseReluDense(forwarded_states)
        forwarded_states = self.decoder_adapter[3 * i + 2](forwarded_states)
        decoder_hidden_states = decoder_hidden_states + self.t5_decoder.block[i].layer[-1].dropout(forwarded_states)
        return decoder_hidden_states

    def compute_layer(self,layer_module,past_key_value,
                      layer_head_mask,cross_attn_layer_head_mask,
                      decoder_hidden_states,encoder_hidden_states,
                      encoder_extended_attention_mask,attention_mask):

        layer_outputs = layer_module(
            decoder_hidden_states,
            attention_mask=attention_mask,
            position_bias=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=None,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=None,
            output_attentions=False,
        )
        decoder_hidden_states = layer_outputs[0]
        return decoder_hidden_states

    def compute_adapter_encoder_output(self,text_input_ids,text_attention_mask):
        inputs_embeds = self.t5_encoder.embed_tokens(text_input_ids)
        encoder_hidden_states = self.t5_encoder.dropout(inputs_embeds)
        past_key_values = [None] * len(self.t5_encoder.block)
        extended_attention_mask = self.t5_model.get_extended_attention_mask(text_attention_mask, text_input_ids.shape)
        head_mask = None
        head_mask = self.t5_encoder.get_head_mask(head_mask, self.t5_encoder.config.num_layers)
        cross_attn_head_mask = None
        cross_attn_head_mask = self.t5_encoder.get_head_mask(cross_attn_head_mask, self.t5_encoder.config.num_layers)

        for i in range(len(self.t5_encoder.block)):
            past_key_value = past_key_values[i]
            layer_head_mask = head_mask[i]
            normed_hidden_states = self.t5_encoder.block[i].layer[0].layer_norm(encoder_hidden_states)
            attention_output = self.t5_encoder.block[i].layer[0].SelfAttention(
                normed_hidden_states,
                mask=extended_attention_mask,
                position_bias=None,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=None,
                output_attentions=False,
            )
            y = attention_output[0]
            y = self.encoder_adapter[2 * i](y)
            encoder_hidden_states = encoder_hidden_states + self.t5_encoder.block[i].layer[0].dropout(y)

            forwarded_states = self.t5_encoder.block[i].layer[-1].layer_norm(encoder_hidden_states)
            forwarded_states = self.t5_encoder.block[i].layer[-1].DenseReluDense(forwarded_states)
            forwarded_states = self.encoder_adapter[2 * i + 1](forwarded_states)
            encoder_hidden_states = encoder_hidden_states + self.t5_encoder.block[i].layer[-1].dropout(forwarded_states)
        encoder_hidden_states = self.t5_encoder.final_layer_norm(encoder_hidden_states)
        encoder_hidden_states = self.t5_encoder.dropout(encoder_hidden_states)
        return encoder_hidden_states

    def compute_encoder_output(self,text_input_ids,text_attention_mask):
        inputs_embeds = self.t5_encoder.embed_tokens(text_input_ids)
        encoder_hidden_states = self.t5_encoder.dropout(inputs_embeds)
        past_key_values = [None] * len(self.t5_encoder.block)
        extended_attention_mask = self.t5_model.get_extended_attention_mask(text_attention_mask, text_input_ids.shape)
        head_mask = None
        head_mask = self.t5_encoder.get_head_mask(head_mask, self.t5_encoder.config.num_layers)
        cross_attn_head_mask = None
        cross_attn_head_mask = self.t5_encoder.get_head_mask(cross_attn_head_mask, self.t5_encoder.config.num_layers)

        for i, (layer_module, past_key_value) in enumerate(zip(self.t5_encoder.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            layer_outputs = layer_module(
                encoder_hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=None,
                output_attentions=False,
            )
            encoder_hidden_states = layer_outputs[0]

        encoder_hidden_states = self.t5_encoder.final_layer_norm(encoder_hidden_states)
        encoder_hidden_states = self.t5_encoder.dropout(encoder_hidden_states)
        return encoder_hidden_states

    def get_decoder_input(self,text_input_ids):
        B = len(text_input_ids)
        decoder_input_ids = torch.ones(B, 1, dtype=torch.long,
                                       device=text_input_ids.device) * self.t5_decoder.config.decoder_start_token_id
        decoder_inputs_embeds = self.t5_decoder.embed_tokens(decoder_input_ids)
        decoder_hidden_states = self.t5_decoder.dropout(decoder_inputs_embeds)
        return decoder_input_ids, decoder_hidden_states

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Grounding Question Answering
        if "gqa" in self.current_tasks:
            ret.update(objectives.compute_gqa(self, batch))

        # Stanford Natural Language Inference - Visual Entailment
        if "snli_ve" in self.current_tasks:
            ret.update(objectives.compute_snli_ve(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        dvp_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss


    def kab_update_step(self):
        if self.kab_val_iter is None:
            val_sampler = DistributedSampler(self.kab_val_dataset, shuffle=True)
            self.kab_val_loader = DataLoader(
                self.kab_val_dataset,
                batch_size = self.hparams.config['batch_size'] // 2,
                sampler = val_sampler,
                num_workers = self.hparams.config['num_workers'],
                collate_fn = self.kab_val_dataset.collate,
            )
            self.kab_val_iter = iter(self.kab_val_loader)
        try:
            batch = next(self.kab_val_iter)

        except StopIteration:
            self.kab_val_iter = iter(self.kab_val_loader)
            batch = next(self.kab_val_iter)

        for i in range(self.search_sample):
            with torch.no_grad():
                output = self(batch)
                if "vqa" in self.current_tasks:
                    loss = output["vqa_loss"]
                if "gqa" in self.current_tasks:
                    loss = output["gqa_loss"]
                if "snli_ve" in self.current_tasks:
                    loss = output["snli_ve_loss"]

                score = output['kab_score']
                self.loss_buffer.append(loss)
                self.score_buffer.append(score)

            arch_loss = 0

            if self.layer_alphas.grad is not None:
                self.layer_alphas.grad.data.zero_()

            for log_probs in self.log_probs_buffer:
                arch_loss = arch_loss + log_probs

            self.log_probs_buffer = []
            arch_loss = -arch_loss
            self.arch_loss_buffer.append(arch_loss)


        ave_loss = sum(self.loss_buffer)/self.search_sample
        ave_arch_loss = sum(self.arch_loss_buffer)/self.search_sample
        ave_score = sum(self.score_buffer)/self.search_sample

        for i in range(self.search_sample):
            reward = self.score_buffer[i]-ave_score
            self.reward_buffer.append(reward)

        for j in range(self.search_sample):
            sample = self.active_index_buffer[j]
            self.layer_alphas.data[sample] += 0.005 * self.reward_buffer[j]*self.up_weight_buffer[j]

        max_reward = max(self.reward_buffer)
        return ave_loss,ave_score,ave_arch_loss,max_reward



    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.step(closure=optimizer_closure)

        if self.search_stage:
            self.kab_update_state = True
            ave_ce_loss, ave_score, ave_arch_loss, max_reward = self.kab_update_step()

            self.kab_print_info += 1
            if self.kab_print_info % 100 == 0 and self.kab_print_info != 0:
                print(
                    'Architecture [%d-%d]\t Arch Loss %.4f\t CE Loss %.4f\t Ave Score %.4f\t Max Reward %.4f\t' % (
                    epoch, batch_idx, ave_arch_loss, ave_ce_loss, ave_score, max_reward))
                print('%s' % (self.get_name()))
                self.kab_print_info = 0

            self.kab_update_state = False
            self.active_index_buffer = []
            self.up_weight_buffer = []
            self.arch_loss_buffer = []
            self.loss_buffer = []
            self.score_buffer = []
            self.reward_buffer = []


    def training_epoch_end(self, outs):
        dvp_utils.epoch_wrapup(self)


    def validation_step(self, batch, batch_idx):
        dvp_utils.set_task(self)
        output = self(batch)

    def get_name(self):
        probs = F.softmax(self.layer_alphas, dim=0).data.cpu().numpy()
        insert_layer = int(np.argmax(probs))
        full_str = 'Insert Layer is ' + str(insert_layer) + ' !'
        return full_str


    def validation_epoch_end(self, outs):
        dvp_utils.epoch_wrapup(self)
        if self.search_stage:
            print('-' * 30 + 'Current Architecture [%d]' % (self.current_epoch) + '-' * 30)
            print('%s' % (self.get_name()))
            print('-' * 60)


    def test_step(self, batch, batch_idx):
        dvp_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        dvp_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return dvp_utils.set_schedule(self)
