from typing import cast, List, Union
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
import torch
from torch import Tensor
import torch.nn.functional as F


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'


def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'


class FlagICLModel:
    def __init__(
        self,
        model_name_or_path: str = None,
        normalize_embeddings: bool = True,
        query_instruction_for_retrieval: str = 'Given a query, retrieval relevant passages that answer the query.',
        examples_for_task: List[dict] = None,
        use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.examples_for_task = examples_for_task

        self.set_examples()
        self.suffix = '\n<response>'

        self.normalize_embeddings = normalize_embeddings

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def set_examples(self, examples_for_task: List[dict] = None):
        if examples_for_task is None and self.examples_for_task is None:
            self.prefix = ''
        elif examples_for_task is not None:
            eg_paris = []
            for i in range(len(examples_for_task)):
                eg_paris.append(
                    get_detailed_example(
                        examples_for_task[i].get('instruct', self.query_instruction_for_retrieval),
                        examples_for_task[i].get('query', ''),
                        examples_for_task[i].get('response', '')
                    )
                )
            self.prefix = '\n\n'.join(eg_paris) + '\n\n'
        else:
            eg_paris = []
            for i in range(len(self.examples_for_task)):
                eg_paris.append(
                    get_detailed_example(
                        self.examples_for_task[i].get('instruct', self.query_instruction_for_retrieval),
                        self.examples_for_task[i].get('query', ''),
                        self.examples_for_task[i].get('response', '')
                    )
                )
            self.prefix = '\n\n'.join(eg_paris) + '\n\n'

    @torch.no_grad()
    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512) -> np.ndarray:
        self.model.eval()
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if isinstance(queries, str):
            sentences = [get_detailed_instruct(self.query_instruction_for_retrieval, queries)]
        else:
            sentences = [get_detailed_instruct(self.query_instruction_for_retrieval, q) for q in queries]

        prefix_ids = self.tokenizer(self.prefix, add_special_tokens=False)['input_ids']
        suffix_ids = self.tokenizer(self.suffix, add_special_tokens=False)['input_ids']

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                max_length=max_length - len(self.tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
                    self.tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
                return_token_type_ids=False,
                truncation=True,
                return_tensors=None,
                add_special_tokens=False
            )
            new_max_length = (len(prefix_ids) + len(suffix_ids) + max_length + 8) // 8 * 8 + 8
            sentences_batch = self.tokenizer.batch_decode(inputs['input_ids'])
            for i in range(len(sentences_batch)):
                sentences_batch[i] = self.prefix + sentences_batch[i] + self.suffix
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=new_max_length,
                add_special_tokens=True
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.float().cpu())

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    @torch.no_grad()
    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        self.model.eval()

        if isinstance(corpus, str):
            sentences = [corpus]
        else:
            sentences = corpus

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=True
            ).to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.float().cpu())

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


class FlagLLMModel:
    def __init__(
        self,
        model_name_or_path: str = None,
        normalize_embeddings: bool = True,
        query_instruction_for_retrieval: str = 'Given a query, retrieval relevant passages that answer the query.',
        use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if isinstance(queries, str):
            input_texts = get_detailed_instruct(self.query_instruction_for_retrieval, queries)
        else:
            input_texts = [get_detailed_instruct(self.query_instruction_for_retrieval, q) for q in queries]
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                pad_to_multiple_of=8,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = last_token_pool(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

from safetensors.torch import load_file
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from torch import nn
import os
import copy
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
        #입력에 어떤 전문가가 적당한지를 계산
        gate_logits = self.gate(inputs)
        #가장 높은 점수를 받은 전문가 num_experts_per_tok만큼 선택
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        #선택된 전문가에 대한 가중치를 softmax 함수를 통해 정규화
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
    
class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True,
            moe: bool = False
    ) -> None:

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if moe:
            self.moe(model_name_or_path, moe)
        else:
            print("moe not")
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
    def moe(self, model_name_or_path, moe):
        state_dict = load_file(os.path.join(model_name_or_path, "model.safetensors"))
        moe_args = MoeArgs(self.model.config.num_experts, self.model.config.num_experts_per_tok, moe)
        for i, layer in enumerate(self.model.encoder.layer):
            if moe == 'intermediate':
                layer.intermediate.dense = MoeLayer(
                    experts=[copy.deepcopy(layer.intermediate.dense) for _ in range(moe_args.num_experts)],
                    gate=torch.nn.Linear(layer.intermediate.dense.in_features, moe_args.num_experts, bias=False),
                    moe_args=moe_args,
                )
            elif moe == 'output':
                layer.output.dense = MoeLayer(
                    experts=[copy.deepcopy(layer.output.dense) for _ in range(moe_args.num_experts)],
                    gate=torch.nn.Linear(layer.output.dense.in_features, moe_args.num_experts, bias=False),
                    moe_args=moe_args,
                )

            for j in range(moe_args.num_experts):
                expert_key = f"encoder.layer.{i}.{moe}.dense.experts.{j}."
                expert_state_dict = {k.replace(expert_key, ""): v for k, v in state_dict.items() if
                                     k.startswith(expert_key)}
                if moe == 'intermediate':
                    layer.intermediate.dense.experts[j].load_state_dict(expert_state_dict)
                elif moe == 'output':
                    layer.output.dense.experts[j].load_state_dict(expert_state_dict)

            # 게이트 네트워크 가중치 로드
            gate_key = f"encoder.layer.{i}.{moe}.dense.gate."
            gate_state_dict = {k.replace(gate_key, ""): v for k, v in state_dict.items() if k.startswith(gate_key)}
            if moe == 'intermediate':
                layer.intermediate.dense.gate.load_state_dict(gate_state_dict)
            elif moe == 'output':
                layer.output.dense.gate.load_state_dict(gate_state_dict)


    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

class LLMEmbedder:
    instructions = {
        "qa": {
            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
    }

    def __init__(
        self,
        model_name_or_path: str = None,
        pooling_method: str = 'cls',
        normalize_embeddings: bool = True,
        use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False

        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 256,
                       task: str = 'qa') -> np.ndarray:
        '''
        Encode queries into dense vectors.
        Automatically add instructions according to given task.
        '''
        instruction = self.instructions[task]["query"]

        if isinstance(queries, str):
            input_texts = instruction + queries
        else:
            input_texts = [instruction + q for q in queries]

        return self._encode(input_texts, batch_size=batch_size, max_length=max_length)

    def encode_keys(self, keys: Union[List[str], str],
                    batch_size: int = 256,
                    max_length: int = 512,
                    task: str = 'qa') -> np.ndarray:
        '''
        Encode keys into dense vectors.
        Automatically add instructions according to given task.
        '''
        instruction = self.instructions[task]["key"]

        if isinstance(keys, str):
            input_texts = instruction + keys
        else:
            input_texts = [instruction + k for k in keys]
        return self._encode(input_texts, batch_size=batch_size, max_length=max_length)

    @torch.no_grad()
    def _encode(self, sentences: Union[List[str], str], batch_size: int = 256, max_length: int = 512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented!")
