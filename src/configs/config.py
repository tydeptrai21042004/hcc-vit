#!/usr/bin/env python3
"""Config system (based on Detectron's)."""

from .config_node import CfgNode

# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = None

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
# one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
_C.MODEL.TRANSFER_TYPE = "linear"
_C.MODEL.WEIGHT_PATH = ""      # if resume from some checkpoint file
_C.MODEL.SAVE_CKPT = False

_C.MODEL.MODEL_ROOT = ""       # root folder for pretrained model weights

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 5
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin:
#   (1) default "random"
#   (2) "final-cls" use aggregated final [cls] embeddings from training dataset
#   (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
#   (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # projection mlp hidden dim
_C.MODEL.PROMPT.DEEP = False  # whether do deep prompt or not, only for prepend location

_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if int -> partial-deep prompt tuning
_C.MODEL.PROMPT.REVERSE_DEEP = False    # if to only update last n layers, not the input layer
_C.MODEL.PROMPT.DEEP_SHARED = False     # if true, all deep layers share the same prompt emb
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # no expand input seq for layers without prompt
# how to get the output emb for cls head:
#   original: follow the original backbone choice
#   img_pool: image patch pool only
#   prompt_pool: prompt embd pool only
#   imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False

# ----------------------------------------------------------------------
# Adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ---- HCC-specific additions (for your Hartley-Cosine adapter) ----
_C.MODEL.ADAPTER.NAME = "none"   # e.g. "HCC" or other adapter names

_C.MODEL.ADAPTER.HCC = CfgNode()
_C.MODEL.ADAPTER.HCC.M = 1
_C.MODEL.ADAPTER.HCC.H = 1                      # legacy single dilation fallback
_C.MODEL.ADAPTER.HCC.AXIS = "hw"               # "h" | "w" | "hw"
_C.MODEL.ADAPTER.HCC.DILATIONS = "1,2,4"       # multi-dilation proposal path; set "1" for old single scale
_C.MODEL.ADAPTER.HCC.SCALE_ADAPTIVE = True      # static/global softmax gate over axis--dilation responses
_C.MODEL.ADAPTER.HCC.SEPARATE_AXIS_KERNELS = True
_C.MODEL.ADAPTER.HCC.GATE_TEMPERATURE = 1.0
_C.MODEL.ADAPTER.HCC.INPUT_ADAPTIVE_GATE = False  # kept only for old CLI compatibility
_C.MODEL.ADAPTER.HCC.GATE_REDUCTION = 4           # ignored in static-gate proposal
_C.MODEL.ADAPTER.HCC.ALPHA_GROUP = 16           # channels per shared alpha group
_C.MODEL.ADAPTER.HCC.PER_CHANNEL = False        # legacy override; True forces ALPHA_GROUP=1
_C.MODEL.ADAPTER.HCC.TIE_SYM = True
_C.MODEL.ADAPTER.HCC.NO_PW = True               # proposal can be ultra-light; USE_PW=True overrides this
_C.MODEL.ADAPTER.HCC.USE_PW = False             # legacy inverse of NO_PW
_C.MODEL.ADAPTER.HCC.PW_RATIO = 32
_C.MODEL.ADAPTER.HCC.PW_GROUPS = 4
_C.MODEL.ADAPTER.HCC.USE_BN = False
_C.MODEL.ADAPTER.HCC.RESIDUAL_SCALE = 1.0
_C.MODEL.ADAPTER.HCC.GATE_INIT = 0.0            # identity-safe inserted adapter
_C.MODEL.ADAPTER.HCC.PADDING = "reflect"
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300

_C.SOLVER.SCHEDULER = "cosine"
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000

_C.SOLVER.DBG_TRAINABLE = False  # if True, will print name of trainable params

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
