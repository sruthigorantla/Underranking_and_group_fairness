#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
cd src/

### for evaluation at top k (--topk) ranks 

# python3 main.py --evaluate german_age25 --topk 20
# python3 main.py --evaluate german_age35 --topk 20

# python3 main.py --evaluate compas_race --topk 100
# python3 main.py --evaluate compas_sex --topk 100

### for evaluation at consecutive (--consecutive) ranks starting from (--start)

# python3 main.py --evaluate german_age25  --consecutive 40 --start 21
# python3 main.py --evaluate german_age35  --consecutive 40 --start 21

# python3 main.py --evaluate compas_race --consecutive 40 --start 21
# python3 main.py --evaluate compas_sex --consecutive 40 --start 21


### for top k evaluation of more than 2 groups 

# python3 main.py --evaluate_multiple_groups german_age --topk 100
# python3 main.py --evaluate_multiple_groups german_age_gender --topk 100


############### with reverse flag

### for evaluation at top k (--topk) ranks 

# python3 main.py --evaluate german_age25 --topk 20 --rev_flag=True
# python3 main.py --evaluate german_age35 --topk 20 --rev_flag=True

# python3 main.py --evaluate compas_race --topk 100 --rev_flag=True

### for evaluation at consecutive (--consecutive) ranks starting from (--start)

# python3 main.py --evaluate german_age25  --consecutive 40 --start 21 --rev_flag=True
# python3 main.py --evaluate german_age35  --consecutive 40 --start 21 --rev_flag=True

# python3 main.py --evaluate compas_race --consecutive 40 --start 21 --rev_flag=True


cd ../