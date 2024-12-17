# Finding Model Features

## Defining a model feature's behavior and impact  
A model's algorithm is made up of a number of model features.
This library uses a class per feature. The class describes a feature in terms of its behavior and impact on predictions.

More specifically, the class includes functions to:
- Declare a tag (aka short name) for the feature e.g. "P18.SC"
- Declare a set of characteristics that identify "candidate" model nodes that _could_ implement the feature. Refer [Filtering](./filter.md) for more detail.
- Implement a strong attribution (ablation) test to confirm whether a candiate node _does_ implement the feature. The test define the expected explicit impact on the model prediction.    

For a given model, this class is applied to the useful nodes to find the actual node(s), if any, that implement the feature.
When the feature is found, the feature tag is added to the node (and is subsequently stored in the JSON file).  

## Example using an Addition Model
Consider a model that performs addition. It can be asked "222222+666966=" and will correctly predict "+0889188".
As detailed in [https://arxiv.org/pdf/2402.02619](https://arxiv.org/pdf/2402.02619) the model's algorithm contains 4 main model features. 

One of those features understands whether a pair of digits (e.g. 2 and 9) sum to 10 or more, and so generates a "carry one".
The model must implement this feature for each of the 6 answer digits. This gives a family of 6 features with names like "A3.SC" 
meaning "this feature impacts the third answer digit and implements the 'sum carry one' task."  

The https://github.com/PhilipQuirke/quanta_maths repository declares the below class for the "carry one" feature.
The class functions have a parameter "impact_digit" so they can define the 6 features with "A0.SC" to "A5.SC" 
```
class add_sc_functions(SubTaskBaseMath):
    
    def tag(impact_digit):
        return An(impact_digit-1)  + ".SC" # e.g. A3.SC

    def prereqs(cfg, position, impact_digit):
        return FilterAnd(
            FilterHead(), # Is an attention head
            FilterPosition(position_name(position)), # Is at token position Px
            FilterPosition(position_name(cfg.num_question_positions+1), QCondition.MIN), # Occurs after the +/- token
            FilterAttention(cfg.dn_to_position_name(attend_digit)), # Attends to Dn
            FilterAttention(cfg.ddn_to_position_name(attend_digit)), # Attends to D'n
            FilterImpact(An(impact_digit))) # Impacts Am
            
    def test(cfg, acfg, impact_digit, strong):
        alter_digit = impact_digit - 1

        intervention_impact = answer_name(impact_digit)

        # 222222 + 666966 = 889188. Has Dn.SC
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(6)]
        store_question[1] += (9 - 6) * (10 ** alter_digit)

        # 333333 + 555555 = 888888. No Dn.SC
        clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(5)]

        # When we intervene we expect answer 889888
        intervened_answer = clean_question[0] + clean_question[1] + 10 ** (alter_digit+1)

        success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        if success:
            print( "Test confirmed", acfg.ablate_node_names, "perform", add_sc_functions.tag(alter_digit), "impacting", intervention_impact, "accuracy.")

        return success
```

## Virtual Features  
If a model feature is implemented by attention heads, the feature is usually implemented in _one_ attention head.
Sometimes a single model feature is implemented by _two_ attention heads in the same layer.
That is the expected outcome of the test function is only achieved when  _two_ attention heads in the same layer are ablated.
The library can automatically handle this case (at the cost of a slower search).
