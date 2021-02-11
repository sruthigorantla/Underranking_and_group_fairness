#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd src/
now=$(date +"%T")
echo "Time before: $now"


########### reranking
## rerank German Credit dataset folds using ALG/FA*IR/CELIS 

# python3 main.py --postprocess german ALG --k 100 
# python3 main.py --postprocess german FAIR 
# python3 main.py --postprocess german CELIS


## rerank COMPAS ProPublica dataset folds using ALG/FA*IR/CELIS 

# python3 main.py --postprocess compas ALG --k 100 
# python3 main.py --postprocess compas FAIR 
# python3 main.py --postprocess compas CELIS

## rerank with more than 2 groups using ALG

python3 main.py --postprocess german ALG --k 100 --multi_group=True



########### reranking with reverse score based ranking as true rank
## rerank German Credit dataset folds using ALG/CELIS 

# python3 main.py --postprocess german ALG --k 100 --rev_flag True
# python3 main.py --postprocess german CELIS --rev_flag True

## rerank COMPAS ProPublica dataset folds using ALG/CELIS 

# python3 main.py --postprocess compas ALG --k 100 --rev_flag True
# python3 main.py --postprocess compas CELIS --rev_flag True


now=$(date +"%T")
echo "Time after: $now"
cd ../