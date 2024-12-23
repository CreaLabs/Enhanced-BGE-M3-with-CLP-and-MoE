# Enhanced-BGE-M3-with-CLP-and-MoE

This repository provides the code for applying Contrastive Learning Penalty (CLP) and Mixture of Experts (MoE) to the BGE-M3 text embedding model for enhanced information retrieval performance.

## Contrastive Learning Penalty (CLP)

CLP is a novel loss function designed to address the limitations of existing contrastive learning methods for improved performance in information retrieval tasks. It incorporates a penalty term that encourages the model to learn more discriminative representations by considering the similarity between negative samples and their corresponding queries.

The CLP loss function is defined as follows:

![CLPL formula](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/imgs/clpl_formula.PNG)

where:

* h<sub>i</sub>: The embedding of the query for the i-th instance.
* h<sub>i</sub><sup>+</sup>: The embedding of the positive sample for the i-th instance.
* H<sup>'</sup>: The set of negative samples for the i-th instance.
* h<sup>'</sup>: The embedding of the negative sample's query.
* H<sup>*</sup>: the set of positive queries for the documents corresponding to the negative samples
* sim(a, b): The cosine similarity function between embeddings a and b.
* τ: The temperature parameter.
* λ: The balancing parameter between the contrastive loss and the penalty term.

The difference between Contrastive Learning Loss and Contrastive Learning Penalty Loss:

![CLP figure](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/imgs/figure1.PNG)

## Specs

- Model

| Model Name | Introduction |
|---|---|
| [bge-m3-ko-CLPL-interMoE](https://huggingface.co/CreaLabs/bge-m3-ko-CLP-interMoE) | This model applies CLPL and MoE, trained on the MIRACL Korean training dataset. MoE is applied to the intermediate layer, and only the MoE layers were trained during fine-tuning. |
| [bge-m3-fa-CLPL-interMoE](https://huggingface.co/CreaLabs/bge-m3-fa-CLP-interMoE) | This model applies CLPL and MoE, trained on the MIRACL Persian training dataset. MoE is applied to the intermediate layer, and only the MoE layers were trained during fine-tuning. |
| [bge-m3-hi-CLPL-interMoE](https://huggingface.co/CreaLabs/bge-m3-hi-CLP-interMoE) | This model applies CLPL and MoE, trained on the MIRACL Hindi  training dataset. MoE is applied to the intermediate layer, and only the MoE layers were trained during fine-tuning. |

- Data
  
Performing negative sampling using the ANCE methodology and generating negative sample's positive queries through the Gemini 1.5 Pro model, which are required for CLPL.

| Dataset | Introduction |
|---|---|
| [ko_CLPL_train_data](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/data/ko_CLP_train_data.jsonl) | MIRACL Korean CLPL training dataset |
| [fa_CLPL_train_data](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/data/fa_CLP_train_data.jsonl) | MIRACL Persian CLPL training dataset |
| [hi_CLPL_train_data](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/data/hi_CLP_train_data.jsonl) | MIRACL Hindi CLPL training dataset |

## Usage

Install:

- train

      git clone https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLPL-and-MoE.git
      pip install -e .
      pip install transformers==4.45.2
      pip install sentencepiece
      pip install protobuf
      pip install simple_parsing

- evalution

      pip install -U FlagEmbedding
      pip install sentencepiece
      pip install protobuf
      pip install faiss-cpu
      pip install faiss-gpu
      pip install nmslib
      pip install pyserini==0.22.1
      pip install peft
      pip install "numpy<2"
      pip install --upgrade datasets
      pip install simple_parsing

Execution:

- train

      python run.py --output_dir CreaLabs/bge-m3-fa-CLPL-outputMoE --model_name_or_path BAAI/bge-m3 --train_data ./train_data --learning_rate 1e-5 --fp16 y --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 128 --passage_max_len 512 --train_group_size 5 --logging_steps 10 --same_task_within_batch True --unified_finetuning False --use_self_distill False --only_train intermediate --moe intermediate --num_experts 2 --num_experts_per_tok 1

- evalution

       python step0-generate_embedding.py --encoder CreaLabs/bge-m3-fa-CLPL-outputMoE --languages ko --index_save_dir ./corpus-index --max_passage_length 8192 --batch_size 4 --fp16 --pooling_method cls --normalize_embeddings True --moe intermediate
       python step1-search_results.py --encoder CreaLabs/bge-m3-fa-CLPL-outputMoE --languages ko fa hi --index_save_dir ./corpus-index --result_save_dir /data/js/search_results --threads 4 --hits 20 --pooling_method cls --normalize_embeddings True --add_instruction False --moe intermediate
       python step2-eval_dense_mldr.py --encoder CreaLabs/bge-m3-fa-CLPL-outputMoE --languages ko --search_result_save_dir ./search_results --qrels_dir ./qrels --eval_result_save_dir ./eval_results --metrics ndcg@5 ndcg@10 --pooling_method cls --normalize_embeddings True


## Evaluation

![Table 2](https://github.com/Dream-Forge-Studios/Enhanced-BGE-M3-with-CLP-and-MoE/blob/main/imgs/table2.PNG)

## Citation

    @misc{
      title={Efficient Fine-tuning Methodology of Text Embedding Models for Information Retrieval: Contrastive Learning Penalty Loss (CLPL)}, 
      author={Jeongsu YU},
      year={2024},
      eprint={},
      archivePrefix={},
      primaryClass={cs.CL}
    }
