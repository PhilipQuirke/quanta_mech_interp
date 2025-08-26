# Quanta Mech Interp

## Introduction
This reuseable library, helps a researcher investigate a model's **algorithm** by:
- searching a model to discover interesting (generic or topic-specific) **model facts** 
- **storing facts** learnt about a model using any Mech Interp (MI) technique in a common json format
- declaring a hypothesised **model feature** as a set of facts (criteria) that must be satisfied 
- searching a model's attention head(s) or an MLP layer that satisfy a model feature's criteria
- storing the features found in the same json format
- declaring a hypothesised **model algorithm** as a set of model features that must exist 
- evaluating a hypothesised algorithm to see if all the required features exist using the model's stored facts

The library includes:
- A technique [Useful Nodes](./useful_tags.md) to reduce the model "search space" to just nodes important to the model's prediction accuracy. 
- A technique [Filtering](./filter.md) to more efficiently [search](./feature.md)  for model facts, features and the overall algorithm. 
- Multiple visualizations of model useful nodes, facts and features e.g.

![Attention](./assets/ins1_mix_d6_l3_h4_t40K_s372001AttentionBehaviorPerHead.svg?raw=true "Attention")

## Folders, Files and Classes 
This library contains files:
- **QuantaMechInterp:** Python library code imported into the notebooks:
  - model_*.py: Contains the configuration of the transformer model being trained/analysed. Includes class ModelConfig 
  - useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions. Includes class UsefulConfig derived from ModelConfig. Refer [Useful_Tags](./useful_tags.md) for more detail. 
  - algo_*.py: Contains tools to support declaring and validating a model algorithm. Includes class AlgoConfig derived from UsefulConfig.
  - quanta_*.py: Contains categorisations of model behavior (aka quanta), with ways to detect, filter and graph them. Refer [Filter](./filter.md) for more detail. 
  - ablate_*.py: Contains ways to "intervention ablate" the model and detect the impact of the ablation

## Installation
From source

```bash
git clone https://github.com/PhilipQuirke/quanta_mech_interp.git
cd QuantaMechInterp
pip install .
```

## Context
This library was created as part of the [Understanding Addition in Transformers](https://arxiv.org/pdf/2310.13121) and [Increasing Trust in Language Models through the Reuse of Verified Circuits](https://arxiv.org/pdf/2402.02619)1
papers. The library is also used by the in-flight "TinySQL" project funded by [WithMartian](https://withmartian.com/)  

