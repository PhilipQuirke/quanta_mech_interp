# model_*.py: Contains the configuration of the transformer model being trained/analysed
from .model_config import ModelConfig
from .model_token_to_char import token_to_char, tokens_to_string
from .model_train import logits_to_tokens_loss, loss_fn, get_training_optimizer_and_scheduler
from .model_train_json import download_huggingface_json, load_training_json
from .model_loss_graph import plot_loss_lines, plot_loss_lines_layout


from .model_sae import AdaptiveSparseAutoencoder, save_sae_to_huggingface 


# useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions
from .useful_config import UsefulConfig 
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, answer_name, NodeLocation, str_to_node_location, UsefulNode, UsefulNodeList


# quanta_*.py: Contains categorisations of model behavior (aka quanta). Applicable to all models
from .quanta_constants import QCondition, QType, MAX_ATTN_TAGS, MIN_ATTN_PERC, NO_IMPACT_TAG, FAIL_SHADES, ATTN_SHADES, ALGO_SHADES, MATH_ADD_SHADES, MATH_SUB_SHADES
from .quanta_file_utils import save_plt_to_file
from .quanta_filter import FilterNode, FilterAnd, FilterOr, FilterName, FilterTrue, FilterHead, FilterNeuron, FilterContains, FilterPosition, FilterLayer, FilterAttention, FilterImpact, FilterAlgo, filter_nodes, print_algo_purpose_results


# ablate_*.py: Contains ways to "intervention ablate" the model and detect the impact of the ablation
from .ablate_config import AblateConfig, acfg
from .ablate_hooks import to_numpy, a_put_resid_post_hook, a_set_ablate_hooks, a_calc_mean_values, a_predict_questions, a_run_attention_intervention
from .ablate_add_useful import ablate_mlp_and_add_useful_node_tags, ablate_head_and_add_useful_node_tags

# model_pca.py: Ways to extract PCA information from model
from .model_pca import calc_pca_for_an, pca_evr_0_percent


# quanta_*.py: Contains ways to detect and graph model behavior (aka quanta) 
from .quanta_add_attn_tags import add_node_attention_tags
from .quanta_map import create_colormap, pale_color, calc_quanta_map
from .quanta_map_attention import get_quanta_attention
from .quanta_map_failperc import get_quanta_fail_perc
from .quanta_map_binary import get_quanta_binary, get_quanta_algo
from .quanta_map_impact import get_answer_impact, get_question_answer_impact, is_answer_sequential, compact_answer_if_sequential, get_quanta_impact, sort_unique_digits


# algo_*.py: Contains utilities to support model algorithm investigation
from .algo_config import AlgoConfig
from .algo_search import search_and_tag_digit_position, search_and_tag_digit, search_and_tag, SubTaskBase



