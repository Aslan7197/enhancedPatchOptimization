from dataclasses import dataclass
from typing import Any

@dataclass()
class Example():
    """A single training/test example."""
    idx: Any
    source: Any
    target: Any

@dataclass()
class InputFeatures():
    """A single training/test features for a example."""
    example_id: Any
    source_ids: Any
    target_ids: Any
    source_mask: Any
    target_mask: Any

@dataclass()
class ModelOption:
    ## Required parameters  
    model_type: str = "roberta"
    model_name_or_path: str = "microsoft/codebert-base"
    tokenizer_name: str = "roberta-base"
    config_name: str = "roberta-base"
    load_model_path: str = None
    input_dir: str = None
    input_filename: str = None
    output_dir: str = None
    output_filename: str = None

    ## Other parameters
    train_filename: str = None
    dev_filename: str = None
    test_filename: str = None

    ## Other parameters
    max_source_length:int = 64
    max_target_length: int= 32
    do_predict: bool = False
    do_train: bool = False
    do_eval: bool = False
    do_test: bool = False
    do_test: bool = False
    do_lower_case: bool = False
    no_cuda: bool = False
    train_batch_size: int = 8
    eval_batch_size:int = 8
    gradient_accumulation_steps: int = 1
    learning_rate:float = 5e-5
    beam_size: int = 50  
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    eval_steps: int = -1
    train_steps: int =-1
    warmup_steps: int = 0
    local_rank: int = -1 
    seed: int = 42

    # Setting Options
    n_gpu: int = 1

    device: str = None

    # Custom Options
    use_special_tokens: bool = True
    use_buggy_contexts: bool = True