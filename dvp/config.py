from sacred import Experiment

ex = Experiment("DVP")

def _loss_names(d):
    ret = {
        "vqa": 0,
        'gqa' : 0,
        'snli_ve':0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "DVP"
    seed = 0
    batch_size = 256
    clip_model = 'ViT-B/16'
    languge_model = None # BERR or T5
    datasets = ["vqa"]
    language_model = 'BERT'
    tokenizer = "bert-base-uncased"
    insert_layer = 0
    loss_names = _loss_names({"vqa": 1})

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224

    # Text Setting
    vqav2_label_size = 3129
    gqa_label_size = 1843
    max_text_len = 16

    # Transformer Setting
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    max_epoch = 10
    max_steps = 2500
    warmup_steps = 0.1

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "./result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # KAB-DILS
    search_sample = 5
    search_stage = False
    use_adapter = False


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "./datasets"
    log_dir = "./result"
    num_gpus = 1
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name

@ex.named_config
def dvp_search_bert_vqa():
    exp_name = "dvp_search_bert_vqa"
    language_model = 'BERT'
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 2
    max_steps = None
    warmup_steps = 0
    learning_rate = 1e-4
    lr_mult = 10
    search_stage = True
    decay_power = 'constant'
    use_adapter = False
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_adaption_bert_vqa():
    exp_name = "dvp_adaption_bert_vqa"
    language_model = 'BERT'
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-4
    lr_mult = 10
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_search_bert_snli_ve():
    exp_name = "dvp_search_bert_snli_ve"
    language_model = 'BERT'
    datasets = ["snli_ve"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 1
    max_steps = None
    warmup_steps = 0
    learning_rate = 1e-4
    search_stage = True
    use_adapter = False
    decay_power = 'constant'
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_adaption_bert_snli_ve():
    exp_name = "dvp_adaption_bert_snli_ve"
    language_model = 'BERT'
    datasets = ["snli_ve"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.2
    learning_rate = 1e-4
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_search_bert_gqa():
    exp_name = "dvp_search_bert_gqa"
    language_model = 'BERT'
    datasets = ["gqa"]
    loss_names = _loss_names({"gqa": 1})
    batch_size = 256
    max_epoch = 2
    max_steps = None
    warmup_steps = 0
    learning_rate = 1e-4
    lr_mult = 10
    search_stage = True
    decay_power = 'constant'
    use_adapter = False
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_adaption_bert_gqa():
    exp_name = "dvp_adaption_bert_gqa"
    language_model = 'BERT'
    datasets = ["gqa"]
    loss_names = _loss_names({"gqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-4
    lr_mult = 10
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "bert-base-uncased"

@ex.named_config
def dvp_search_t5_vqa():
    exp_name = "dvp_search_t5_vqa"
    language_model = 'T5'
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 2
    max_steps = None
    warmup_steps = 0
    learning_rate = 2e-4
    lr_mult = 10
    search_stage = True
    decay_power = 'constant'
    use_adapter = False
    tokenizer = "t5-base"

@ex.named_config
def dvp_adaption_t5_vqa():
    exp_name = "dvp_adaption_t5_vqa"
    language_model = 'T5'
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-4
    lr_mult = 10
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "t5-base"

@ex.named_config
def dvp_search_t5_snli_ve():
    exp_name = "dvp_search_t5_snli_ve"
    language_model = 'T5'
    datasets = ["snli_ve"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 1
    max_steps = None
    warmup_steps = 0
    learning_rate = 2e-4
    search_stage = True
    use_adapter = False
    decay_power = 'constant'
    tokenizer = "t5-base"

@ex.named_config
def dvp_adaption_t5_snli_ve():
    exp_name = "dvp_adaption_t5_snli_ve"
    language_model = 'T5'
    datasets = ["snli_ve"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.2
    learning_rate = 2e-4
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "t5-base"

@ex.named_config
def dvp_search_t5_gqa():
    exp_name = "dvp_search_t5_gqa"
    language_model = 'T5'
    datasets = ["gqa"]
    loss_names = _loss_names({"gqa": 1})
    batch_size = 256
    max_epoch = 2
    max_steps = None
    warmup_steps = 0
    learning_rate = 2e-4
    lr_mult = 10
    search_stage = True
    decay_power = 'constant'
    use_adapter = False
    tokenizer = "t5-base"

@ex.named_config
def dvp_adaption_t5_gqa():
    exp_name = "dvp_adaption_t5_gqa"
    language_model = 'T5'
    datasets = ["gqa"]
    loss_names = _loss_names({"gqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 2e-4
    lr_mult = 10
    insert_layer = 0
    search_stage = False
    use_adapter = False
    tokenizer = "t5-base"
