# Node Filters

Model features are implemented by model nodes (attention heads and MLP layers).
How do we map from a hypothesised model feature to a set of actual model nodes?
We start by declaring a set of expected facts that characterise and identify the model fature.
Many facts can be declared as "node filters". 
When searching a specific model for a hypothesised feature, these filters reduce the search space size, saving time and money.      

## Example 
Suppose we have an arithmetic addition model, and are searching for a feature that has these characteristics:
- Is at token position P14
- Is implemented using attention heads
- Attends to (at least) D2 and D'2
- Impacts (at least) answer token A3

In a Colab, we can find search the model's useful nodes (cfg.useful_nodes) for this feature using this filter:

````
import QuantaMechInterp as qt

my_filters = qt.FilterAnd(
    qt.FilterHead(),  # Is an attention head
    qt.FilterPosition(qt.position_name(14)), # Is at token position 14
    qt.FilterAttention(cfg.dn_to_position_name(2)), # Attends to D2
    qt.FilterAttention(cfg.ddn_to_position_name(2)), # Attends to D'2
    qt.FilterImpact(qt.answer_name(3))) # Impacts third answer token

test_nodes = qt.filter_nodes(cfg.useful_nodes, my_filters)
````

Filters find candidate nodes that _could_ implement a specific model feature. 
Confirming that a node _does_ implement the model feature may require an additional context-specific test.

## Filter Types
The available Filters are:
- FilterAnd: node must satisfy all the child criteria to be selected 
- FilterOr: node must satisfy at least one child criteria to be selected
- FilterHead: node must be an attention head
- FilterNeuron: node must be an MLP neuron
- FilterPosition: node must be located the the specified location
- FilterContains: node tags must contain the specified text
- FilterAttention: node must attend to the specified token and (optionally) with at least the specified percentage strength 
- FilterImpact: node must impact the specified answer token(s)
- FilterAlgo: node must have the specified algorithm tag

The library can be extended with topic-additional specific filter classes as desired.
