# Quanta Mech Interp

This reuseable library, helps a researcher investigate a model's **algorithm** by:
- searching a model to discover interesting (generic or topic-specific) **model facts** 
- **storing facts** learnt about a model using any Mech Interp (MI) technique in a common json format
- declaring a hypothesised **model feature** as a set of "criteria" facts must be satisfied 
- searching a model's attention head(s) or an MLP layer that satisfy a model feature's criteria
- storing the features found in the same json format
- declaring a hypothesised **model algorithm** as a set of model features that must exist 
- evaluate a hypothesised algorithm against a model's stored facts

This library was created as part of the [Understanding Addition in Transformers](https://arxiv.org/pdf/2310.13121) and [Increasing Trust in Language Models through the Reuse of Verified Circuits](https://arxiv.org/pdf/2402.02619)1
papers. The library is also used by the in-flight "TinySQL" project funded by [WithMartian](https://withmartian.com/)  
