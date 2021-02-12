# On the Problem of Underranking in Group-Fair Ranking.
This repository consists of code and datasets for our paper. We extend the code of [FA\*IR](https://github.com/fair-search) and implement the DP algorithm in [Celis et al. - Ranking with Fairness Constraints](https://arxiv.org/pdf/1704.06840.pdf), to compare our work with these algorithms as baselines. 


## Requirements
``python``, ``numpy``, ``matplotlib``


### Re-ranking for fairness 

The root directory contains the following bash-script for re-ranking with ALG and also the baselines FA\*IR (FAIR) and Celis et al (CELIS).

``postprocess.sh`` 


#### To run the experiments

``python3 main.py --postprocess <dataset> <method> --k <top k ranks> --rev_flag <True/False> --multi_group <True/False>`` re-ranks the true ranking based on the method (ALG/FAIR/CELIS) and writes the re-ranked data in the folder ``data/<dataset>/..._<experiment>.txt``. If ``rev_flag`` is set, the true ranking is based on the negative scores (or relavance). The flag ``multi_group`` is set for the experiments on the German Credit dataset with more than 2 groups.

Uncomment the experiment required and use ``./postprocess.sh``
 

### Result Evaluation

The following bash-script evaluates results on all the datasets.

``evaluateResults.sh``

#### Two types of evaluations

1. Top k evaluation,
``python3 main.py --evaluate <dataset_protectedgroup> --topk <top k ranks> --rev_flag <True/False>`` evaluates the algorithms for the top k ranks, where k is given as input argument ``topk``. The usage of ``ref_flag`` is same as above.
2. Evaluation for k consecutive ranks
``python3 main.py --evaluate german_age25  --consecutive 40 --start 21 --rev_flag <True/False>`` evaluates the algorithms for consecutive ranks (use the argument ``consecutive`` to choose how many consecutive ranks to evaluate for), starting from rank given as ``start``. The usage of ``ref_flag`` is same as above.
