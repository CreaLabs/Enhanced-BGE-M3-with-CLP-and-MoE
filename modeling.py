import logging
from dataclasses import dataclass
from typing import Dict, Optional
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.file_utils import ModelOutput
from huggingface_hub import snapshot_download
from typing import List
from simple_parsing.helpers import Serializable
import copy
import pickle
from transformers import XLMRobertaConfig, XLMRobertaModel

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BGEM3Model(nn.Module):

    def __init__(self,
                 model_name: str = None,
                 config: AutoConfig = None,
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 enable_sub_batch: bool = True,
                 unified_finetuning: bool = True,
                 use_self_distill: bool = False,
                 colbert_dim: int = -1,
                 self_distill_start_step: int = -1,
                 moe: str = '',
                 only_train: str = '',
                 num_experts: int = 1,
                 num_experts_per_tok: int = 1,
                 ):
        super().__init__()
        self.config = config

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe = moe
        self.only_train = only_train

        self.load_model(model_name, colbert_dim=colbert_dim)
        self.vocab_size = self.model.config.vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.unified_finetuning = unified_finetuning
        if not self.unified_finetuning:
            self.colbert_linear = None
            self.sparse_linear = None

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.enable_sub_batch = enable_sub_batch
        self.temperature = temperature
        self.use_self_distill = use_self_distill
        self.self_distill_start_step = self_distill_start_step

        self.step = 0
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # self.adapter = nn.Sequential(
        #     nn.Linear(config.hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, config.hidden_size)
        # )

    def load_model(self, model_name, colbert_dim: int = -1):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])

        self.model = AutoModel.from_pretrained(model_name)
        if self.moe:
            moe_args = MoeArgs(self.num_experts, self.num_experts_per_tok, self.moe)
            for layer in self.model.encoder.layer:
                if self.moe == 'output':
                    layer.output.dense = MoeLayer(
                        experts=[copy.deepcopy(layer.output.dense) for _ in range(self.num_experts)],
                        gate=torch.nn.Linear(layer.output.dense.in_features, self.num_experts, bias=False),
                        moe_args=moe_args,
                    )
                    print("moe output")
                elif self.moe == 'intermediate':
                    layer.intermediate.dense = MoeLayer(
                        experts=[copy.deepcopy(layer.intermediate.dense) for _ in range(self.num_experts)],
                        gate=torch.nn.Linear(layer.intermediate.dense.in_features, self.num_experts, bias=False),
                        moe_args=moe_args,
                    )
                    print("moe intermediate")
        else:
            print("not moe")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.colbert_linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                              out_features=self.model.config.hidden_size if colbert_dim == -1 else colbert_dim)
        self.sparse_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=1)

        if os.path.exists(os.path.join(model_name, 'colbert_linear.pt')) and os.path.exists(
                os.path.join(model_name, 'sparse_linear.pt')):
            logger.info('loading existing colbert_linear and sparse_linear---------')
            self.load_pooler(model_dir=model_name)
        else:
            logger.info(
                'The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def neg_q_dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = torch.diag(scores)  # 대각 행렬 추출
        return scores

    def sparse_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def colbert_score(self, q_reps, p_reps, q_mask: torch.Tensor):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        scores = scores / self.temperature
        return scores

    def _encode(self, features):
        dense_vecs, sparse_vecs, colbert_vecs = None, None, None
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state

        # last_hidden_state = self.adapter(last_hidden_state)

        dense_vecs = self.dense_embedding(last_hidden_state, features['attention_mask'])
        if self.unified_finetuning:
            sparse_vecs = self.sparse_embedding(last_hidden_state, features['input_ids'])
            colbert_vecs = self.colbert_embedding(last_hidden_state, features['attention_mask'])
        if self.normlized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)
        return dense_vecs, sparse_vecs, colbert_vecs

    def encode(self, features, sub_batch_size=None):
        if features is None:
            return None

        if sub_batch_size is not None and sub_batch_size != -1:
            all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
            for i in range(0, len(features['attention_mask']), sub_batch_size):
                end_inx = min(i + sub_batch_size, len(features['attention_mask']))
                sub_features = {}
                for k, v in features.items():
                    sub_features[k] = v[i:end_inx]

                dense_vecs, sparse_vecs, colbert_vecs = self._encode(sub_features)
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)
                all_colbert_vecs.append(colbert_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)
        else:
            dense_vecs, sparse_vecs, colbert_vecs = self._encode(features)

        if self.unified_finetuning:
            return dense_vecs.contiguous(), sparse_vecs.contiguous(), colbert_vecs.contiguous()
        else:
            return dense_vecs.contiguous(), None, None

    def compute_sub_batch_size(self, features):
        mapping = [(6000, 1), (5000, 2), (4000, 3), (3000, 3), (2000, 5), (1000, 9), (512, 16), (0, 32)]
        cur_l = features['input_ids'].size(-1)
        for l, b in mapping:
            if cur_l >= l:
                return b

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def distill_loss(self, teacher_targets, student_scores, group_size):
        labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
        labels = labels * group_size

        loss = 0
        mask = torch.zeros_like(student_scores)
        for i in range(group_size):
            temp_target = labels + i
            temp_scores = student_scores + mask
            temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")  # B
            loss += torch.mean(teacher_targets[:, i] * temp_loss)
            mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                 value=torch.finfo(student_scores.dtype).min)
        return loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_scores: Tensor = None,
                bi_directions=None, neg_q: Dict[str, Tensor] = None):
        if self.enable_sub_batch:
            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(query,
                                                                      sub_batch_size=self.compute_sub_batch_size(query))
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passage,
                                                                      sub_batch_size=self.compute_sub_batch_size(
                                                                          passage))
            if neg_q:
                neg_q_dense_vecs, neg_q_sparse_vecs, neg_q_colbert_vecs = self.encode(neg_q,
                                                                                      sub_batch_size=self.compute_sub_batch_size(
                                                                                          passage))
        else:
            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(query)
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passage)
            if neg_q:
                neg_q_dense_vecs, neg_q_sparse_vecs, neg_q_colbert_vecs = self.encode(neg_q)

        if self.training:
            if teacher_scores is not None:
                # print("Use soft-label distillation...")
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # B N
                group_size = p_sparse_vecs.size(0) // q_sparse_vecs.size(0)

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)
                    cross_teacher_targets = self._dist_gather_tensor(teacher_targets)
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.distill_loss(cross_teacher_targets, cross_dense_scores, group_size=group_size)
                else:
                    loss = self.distill_loss(teacher_targets, dense_scores, group_size=group_size)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.distill_loss(teacher_targets, sparse_scores, group_size=group_size)

                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=query['attention_mask'])  # B, B * N
                    colbert_loss = self.distill_loss(teacher_targets, colbert_scores, group_size=group_size)

                    ensemble_loss = self.distill_loss(teacher_targets,
                                                      dense_scores + 0.3 * sparse_scores + colbert_scores,
                                                      group_size=group_size)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4


            else:
                idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)
                targets = idxs * (p_dense_vecs.size(0) // q_dense_vecs.size(0))

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)

                    cross_idxs = torch.arange(cross_q_dense_vecs.size(0), device=cross_q_dense_vecs.device,
                                              dtype=torch.long)

                    cross_targets = cross_idxs * (cross_p_dense_vecs.size(0) // cross_q_dense_vecs.size(0))
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.compute_loss(cross_dense_scores, cross_targets)
                else:
                    loss = self.compute_loss(dense_scores, targets)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.compute_loss(sparse_scores, targets)

                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=query['attention_mask'])  # B, B * N
                    colbert_loss = self.compute_loss(colbert_scores, targets)

                    ensemble_loss = self.compute_loss(dense_scores + 0.3 * sparse_scores + colbert_scores, targets)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4

            if self.use_self_distill and self.step > self.self_distill_start_step and self.unified_finetuning:
                ensemble_scores = dense_scores + 0.3 * sparse_scores + colbert_scores
                teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
                ensemble_distill_dense_loss = - torch.mean(
                    torch.sum(torch.log_softmax(dense_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_sparse_loss = - torch.mean(
                    torch.sum(torch.log_softmax(sparse_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_colbert_loss = - torch.mean(
                    torch.sum(torch.log_softmax(colbert_scores, dim=-1) * teacher_targets, dim=-1))
                loss += (
                                    ensemble_distill_dense_loss + 0.1 * ensemble_distill_sparse_loss + ensemble_distill_colbert_loss) / 3
                loss = loss / 2
            if neg_q:
                neg_q_sim = F.cosine_similarity(neg_q_dense_vecs, p_dense_vecs[1:], dim=-1)
                neg_q_loss = (1 - neg_q_sim).mean()
                lambda_func = 0.1
                loss = loss * (1 - lambda_func) + lambda_func * neg_q_loss
            self.step += 1
        else:
            loss = None
        return EncoderOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict


        if self.moe or self.only_train:
            config = self.model.config
            if self.moe:
                config.update({
                    "num_experts": self.num_experts,
                    "num_experts_per_tok": self.num_experts_per_tok,
                    "only_train": self.only_train,
                    "moe": self.moe
                })
            else:
                config.update({
                    "only_train": self.only_train,
                })
            config.save_pretrained(output_dir)

        moe_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(moe_state_dict))  # MoE 가중치 저장


        if self.unified_finetuning:
            torch.save(_trans_state_dict(self.colbert_linear.state_dict()),
                       os.path.join(output_dir, 'colbert_linear.pt'))
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()),
                       os.path.join(output_dir, 'sparse_linear.pt'))

    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)


@dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int
    moe: str


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        # 입력에 어떤 전문가가 적당한지를 계산
        gate_logits = self.gate(inputs)
        # 가장 높은 점수를 받은 전문가 num_experts_per_tok만큼 선택
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        # 선택된 전문가에 대한 가중치를 softmax 함수를 통해 정규화
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(inputs.dtype)

        if self.args.moe == 'output':
            results = torch.zeros(inputs.size(0), inputs.size(1), 1024, device=inputs.device, dtype=inputs.dtype)
        elif self.args.moe == 'intermediate':
            results = torch.zeros(inputs.size(0), inputs.size(1), 4096, device=inputs.device, dtype=inputs.dtype)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_token, nth_expert = torch.where(selected_experts == i)
            results[batch_idx, nth_token] += weights[batch_idx, nth_token, nth_expert, None] * expert(
                inputs[batch_idx, nth_token]
            )
        return results


class BGEM3ForInference(BGEM3Model):

    def forward(self,
                text_input: Dict[str, Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert: bool = False,
                return_sparse_embedding: bool = False):
        assert return_dense or return_sparse or return_colbert, 'Must choose one or more from `return_colbert`, `return_sparse`, `return_dense` to set `True`!'

        last_hidden_state = self.model(**text_input, return_dict=True).last_hidden_state

        output = {}
        if return_dense:
            dense_vecs = self.dense_embedding(last_hidden_state, text_input['attention_mask'])
            output['dense_vecs'] = dense_vecs
        if return_sparse:
            sparse_vecs = self.sparse_embedding(last_hidden_state, text_input['input_ids'],
                                                return_embedding=return_sparse_embedding)
            output['sparse_vecs'] = sparse_vecs
        if return_colbert:
            colbert_vecs = self.colbert_embedding(last_hidden_state, text_input['attention_mask'])
            output['colbert_vecs'] = colbert_vecs

        if self.normlized:
            if 'dense_vecs' in output:
                output['dense_vecs'] = torch.nn.functional.normalize(output['dense_vecs'], dim=-1)
            if 'colbert_vecs' in output:
                output['colbert_vecs'] = torch.nn.functional.normalize(output['colbert_vecs'], dim=-1)

        return output