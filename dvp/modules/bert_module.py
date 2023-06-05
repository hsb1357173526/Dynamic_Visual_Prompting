import torch
import torch.nn as nn
import pytorch_lightning as pl
from dvp.modules import objectives, dvp_utils
import clip
from transformers import BertConfig, BertModel
import math
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Define cross attention layer
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

# Define adapter
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

# Define BERT model
class DVP_BERT(pl.LightningModule):
    def __init__(self, config, kab_val_dataset=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = 'BERT'
        self.text_length = config["max_text_len"]

        # Load BERT model
        self.bert_config = BertConfig.from_pretrained(config['tokenizer'])
        self.bert_model = BertModel.from_pretrained(config['tokenizer'], config=self.bert_config)
        self.bert_embeddings = self.bert_model.embeddings
        self.bert_encoder = self.bert_model.encoder
        self.position_ids = torch.arange(self.text_length).expand((1, -1))

        # Load CLIP visual encoder
        self.visumodel = clip.load(config["clip_model"], jit=False, device=torch.device("cpu"))[0].visual

        self.search_stage = config['search_stage']
        self.use_adapter = config['use_adapter']

        # Use parameter-efficient training, only train LayerNorm layers and adapter
        if self.use_adapter:

            self.adapter_modules = nn.ModuleList(
                [Adapter(config["hidden_size"]) for _ in range(2 * config['num_layers'])]
            )

            for name, param in self.bert_embeddings.named_parameters():
                param.requires_grad = False

            for name, param in self.bert_encoder.named_parameters():
                if 'LayerNorm' not in name:
                    param.requires_grad = False


        if self.search_stage:

            self.search_sample = config['search_sample']

            # Define cross attention layer and token_type_embedding candidates of KAB-APP
            self.token_type_embeddings_options = nn.ModuleList([nn.Embedding(2, config["hidden_size"])
                                                                for _ in range(config["num_layers"])])

            self.cross_attn_options = nn.ModuleList([Cross_Attention(config["hidden_size"],
                                                                     config["drop_rate"],
                                                                     config["num_heads"])
                                                     for _ in range(config["num_layers"])])

            # For KAB-APP mini-val
            self.kab_val_dataset = kab_val_dataset
            self.kab_val_iter = None

            # Define preference of all possible layers
            self.layer_alphas = nn.Parameter(torch.zeros(config["num_layers"]))

            # Define whether in KAB-APP samping stage
            self.kab_update_state = False

            # Information of every KAB-APP sampling
            self.active_index_buffer = []
            self.up_weight_buffer = []
            self.log_probs_buffer = []
            self.arch_loss_buffer = []
            self.loss_buffer = []
            self.score_buffer = []
            self.reward_buffer = []

            # Frequency of printing KAB-APP searched result
            self.kab_print_info = 0

        else:
            # Use KAB-APP searched result to insert DVP for adapting V&L tasks
            self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
            self.cross_attn = Cross_Attention(config["hidden_size"],config["drop_rate"],config["num_heads"])
            self.insert_layer = config['insert_layer']

        # Frozen visual encoder
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
        image_token_type_idx = 1
    ):
        # Extract image and text from batch
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
        image = batch[imgkey].cuda(torch.cuda.current_device())
        text_input_ids = batch[f"text_input_ids"].cuda(torch.cuda.current_device())
        text_token_type_ids = batch[f"text_token_type_ids"].cuda(torch.cuda.current_device())
        text_attention_mask = batch[f"text_attention_mask"].cuda(torch.cuda.current_device())

        # Extract image feature
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

        # Text Embedding
        past_key_values_length = 0
        embedding_output = self.bert_embeddings(
            input_ids=text_input_ids,
            position_ids=self.position_ids.to(text_input_ids.device),
            token_type_ids=text_token_type_ids,
            inputs_embeds=None,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = embedding_output

        # Define attention mask
        prompt_attention_mask = torch.ones(x.shape[0], 1).cuda()
        attention_mask = self.get_mask(text_attention_mask)
        merge_attention_mask = self.get_merge_attention_mask(prompt_attention_mask, text_attention_mask)


        if self.training:

            if self.search_stage:

                # KAB-APP randomly choose a insertion layer to train
                if not self.kab_update_state:
                    insert_layer = random.randint(0, len(self.layer_alphas) - 1)

                # KAB-APP choose some layers based on perferences in sampling stage
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


            for i in range(len(self.bert_encoder.layer)):

                # Before inserting DVP, we process text feature in anterior layers
                if i < insert_layer:
                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, attention_mask)

                elif i == insert_layer:
                    text_cls_token = encoder_outputs[:, :1, :]

                    if self.search_stage:
                        # KAB-APP use corresponding cross attention layer candidates of sampled result to generate DVP
                        visual_prompt = self.cross_attn_options[i](x, x, text_cls_token, None)

                        encoder_outputs, visual_prompt = (encoder_outputs + self.token_type_embeddings_options[i](torch.zeros_like(text_input_ids, device=text_input_ids.device).long()),
                                                          visual_prompt + self.token_type_embeddings_options[i](torch.full_like(torch.ones(visual_prompt.shape[0], visual_prompt.shape[1]),image_token_type_idx).long().to(text_input_ids.device)))
                    else:
                        # Normal training
                        visual_prompt = self.cross_attn(x, x, text_cls_token, None)
                        encoder_outputs, visual_prompt = (encoder_outputs + self.token_type_embeddings(
                            torch.zeros_like(text_input_ids, device=text_input_ids.device).long()),
                                                          visual_prompt + self.token_type_embeddings(
                                                              torch.full_like(torch.ones(visual_prompt.shape[0],
                                                                                         visual_prompt.shape[1]),
                                                                              image_token_type_idx).long().to(
                                                                  text_input_ids.device)))
                    # Concatenate DVP and text features
                    encoder_outputs = torch.cat([visual_prompt, encoder_outputs], dim=1)
                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, merge_attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, merge_attention_mask)

                # Subsequent layers after inserting DVP
                else:
                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, merge_attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, merge_attention_mask)


        # If we're in val or test stage
        else:
            if self.search_stage:
                # Choose the layer of maximum preference
                probs = F.softmax(self.layer_alphas, dim=0).data.cpu().numpy()
                insert_layer = int(np.argmax(probs))

            else:
                insert_layer = self.insert_layer

            for i in range(len(self.bert_encoder.layer)):

                if i < insert_layer:
                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, attention_mask)

                elif i == insert_layer:
                    text_cls_token = encoder_outputs[:, :1, :]
                    if self.search_stage:
                        visual_prompt = self.cross_attn_options[i](x, x, text_cls_token, None)
                        encoder_outputs, visual_prompt = (encoder_outputs + self.token_type_embeddings_options[i](
                            torch.zeros_like(text_input_ids, device=text_input_ids.device).long()),
                                                          visual_prompt + self.token_type_embeddings_options[i](
                                                              torch.full_like(torch.ones(visual_prompt.shape[0],
                                                                                         visual_prompt.shape[1]),
                                                                              image_token_type_idx).long().to(
                                                                  text_input_ids.device)))
                    else:
                        visual_prompt = self.cross_attn(x, x, text_cls_token, None)
                        encoder_outputs, visual_prompt = (encoder_outputs + self.token_type_embeddings(
                            torch.zeros_like(text_input_ids, device=text_input_ids.device).long()),
                                                          visual_prompt + self.token_type_embeddings(
                                                              torch.full_like(torch.ones(visual_prompt.shape[0],
                                                                                         visual_prompt.shape[1]),
                                                                              image_token_type_idx).long().to(
                                                                  text_input_ids.device)))

                    encoder_outputs = torch.cat([visual_prompt, encoder_outputs], dim=1)

                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, merge_attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, merge_attention_mask)

                else:
                    if not self.use_adapter:
                        encoder_outputs = self.compute_layer(i, encoder_outputs, merge_attention_mask)
                    else:
                        encoder_outputs = self.compute_adapter_layer(i, encoder_outputs, merge_attention_mask)

        # Using [CLS] token to connect classifier
        cls_feats = encoder_outputs[:, 1, :]

        ret = {
            "cls_feats": cls_feats,
        }

        return ret


    # Add adapter to BERT's layers
    def compute_adapter_layer(self,index,encoder_outputs,attention_mask):
        self_outputs = self.bert_encoder.layer[index].attention.self(
            hidden_states=encoder_outputs,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=None)
        self_outputs = self_outputs[0]
        attention_output = self.bert_encoder.layer[index].attention.output.dense(self_outputs)
        attention_output = self.bert_encoder.layer[index].attention.output.dropout(attention_output)
        attention_output = self.bert_encoder.layer[index].attention.output.LayerNorm(
            encoder_outputs + self.adapter_modules[2 * index](attention_output))

        intermediate_output = self.bert_encoder.layer[index].intermediate(attention_output)
        intermediate_output = self.bert_encoder.layer[index].output.dense(intermediate_output)
        intermediate_output = self.bert_encoder.layer[index].output.dropout(intermediate_output)
        encoder_outputs = self.bert_encoder.layer[index].output.LayerNorm(
            attention_output + self.adapter_modules[2 * index + 1](intermediate_output))
        return encoder_outputs


    def compute_layer(self,index,encoder_outputs,attention_mask):
        encoder_outputs = self.bert_encoder.layer[index](
            encoder_outputs,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
        )
        encoder_outputs = encoder_outputs[0]
        return encoder_outputs


    # Get text mask
    def get_mask(self, text_attention_mask):
        B, L = text_attention_mask.shape
        extended_attention_mask = text_attention_mask.unsqueeze(1).repeat(1, L, 1)
        extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1,
                                                                              self.bert_config.num_attention_heads,
                                                                              1,
                                                                              1)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.bert_model.dtype).min
        return extended_attention_mask

    # Get DVP mask concatenated with text mask
    def get_merge_attention_mask(self, prompt_attention_mask, text_attention_mask):
        extended_attention_mask = torch.cat([prompt_attention_mask, text_attention_mask], dim=-1)
        B, L = extended_attention_mask.shape
        extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1, L, 1)
        extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1, self.bert_config.num_attention_heads, 1, 1)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.bert_model.dtype).min
        return extended_attention_mask

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

        # Load KAB-APP mini-val dataloader
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

        # Sampling 5 times
        for i in range(self.search_sample):
            with torch.no_grad():
                output = self(batch)
                # Calculate loss for current sampled result
                if "vqa" in self.current_tasks:
                    loss = output["vqa_loss"]
                if "gqa" in self.current_tasks:
                    loss = output["gqa_loss"]
                if "snli_ve" in self.current_tasks:
                    loss = output["snli_ve_loss"]

                # Score for current sampled result
                score = output['kab_score']

                self.loss_buffer.append(loss)
                self.score_buffer.append(score)

            arch_loss = 0

            if self.layer_alphas.grad is not None:
                self.layer_alphas.grad.data.zero_()

            # Calculate architecture loss
            for log_probs in self.log_probs_buffer:
                arch_loss = arch_loss + log_probs
            self.log_probs_buffer = []
            arch_loss = -arch_loss
            self.arch_loss_buffer.append(arch_loss)


        ave_loss = sum(self.loss_buffer) / self.search_sample
        ave_arch_loss = sum(self.arch_loss_buffer) / self.search_sample
        ave_score = sum(self.score_buffer) / self.search_sample

        # Calculate reword for sampled layer
        for i in range(self.search_sample):
            reward = self.score_buffer[i]-ave_score
            self.reward_buffer.append(reward)

        # Update prefences
        for j in range(self.search_sample):
            sample = self.active_index_buffer[j]
            self.layer_alphas.data[sample] += 0.005 * self.reward_buffer[j]*self.up_weight_buffer[j]


        max_reward = max(self.reward_buffer)

        return ave_loss,ave_score,ave_arch_loss,max_reward



    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        optimizer.step(closure=optimizer_closure)

        # If search stage is True, we will start KAB-APP sampling after randomly sampling for training
        if self.search_stage:
            # While sampling, the state is set to True
            self.kab_update_state = True

            # Sampling and updating preference
            ave_ce_loss, ave_score, ave_arch_loss, max_reward = self.kab_update_step()

            # Print info of KAB-APP
            self.kab_print_info += 1
            if self.kab_print_info % 100 == 0 and self.kab_print_info != 0:
                print('Architecture [%d-%d]\t Arch Loss %.4f\t CE Loss %.4f\t Ave Score %.4f\t Max Reward %.4f\t' %(epoch, batch_idx, ave_arch_loss, ave_ce_loss, ave_score, max_reward))
                print('%s' % (self.get_name()))
                self.kab_print_info = 0

            # End sampling and clear sampling info, start randomly sampling insert layer for training
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

    # Print the result with maximum preference
    def get_name(self):
        probs = F.softmax(self.layer_alphas, dim=0).data.cpu().numpy()
        insert_layer = int(np.argmax(probs))
        full_str = 'Insert Layer is '+str(insert_layer) +' !'
        return full_str



    def validation_epoch_end(self, outs):
        dvp_utils.epoch_wrapup(self)

        # When one epoch is finishe, we'll print the current searched result
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
