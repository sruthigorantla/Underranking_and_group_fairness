import processingWithFair.fair.post_processing_methods.fair_ranker.create as fair
from processingWithFair.fair.dataset_creator.candidate import Candidate


from processingWithFair.metrics import precision_at
from ALG.gfair_underranking import gfair_underranking
from CELIS.celis_et_al import ConstrainedDP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rerank_alg(dataDescription, dataset, p_deviation=None, iteration=None, k=100, rev=False):

    if 'german' in dataset:
        # print(dataDescription.orig_data_path)
        # print(dataDescription.header)
        # DurationMonth,CreditAmount,score,age25
        data = pd.read_csv(dataDescription.orig_data_path)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print("German Credit p value is: ",p)
        
        # NUM_GROUPS = len(data.groupby(dataDescription.protected_group).count())
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        # for German credit, COMPAS and other such datasests, add query_id as all 1s.
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        
        # sort the training data based on true scores
        
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        
        # add 'doc_id' here itself
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')

    elif 'biased_normal' in dataset:
        # dataDescription.header  ----   DurationMonth,CreditAmount,score,age25
        data = pd.read_csv(dataDescription.orig_data_path)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print("Biased Normal Synthetic p value is: ",p)
        
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        # for German credit, COMPAS and other such datasests, add query_id as all 1s.
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        
        # sort the training data based on true scores
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        
        # add 'doc_id' here itself
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')
    
      
    
    elif 'compas' in dataset:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print("COMPAS p value is: ",p)
        
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        if rev:
            data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        else:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
            data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')

    
    np_data = np.array(data)
    try:
        true_ranking = [f"id-{int(x)}" for x in data["doc_id"]] # 2nd column is ranking
    except KeyError:
        true_ranking = [f"id-{int(x+1)}" for x in range(len(data))]
    
    id_2_group = {}
    id_2_row = {}
    for idx, (id, group) in enumerate(zip(true_ranking, data["prot_attr"])):
        id_2_group[id] = int(group)
        id_2_row[id] = np_data[idx]
    
    # Now run the algo
    
    ##### GFair and Underranking version
    
    
    final_ranking = gfair_underranking(true_ranking, id_2_group, NUM_GROUPS, p, p_deviation, k, rev)

    

    # Contruct the dataframe back for dumping purposes
    length = len(final_ranking)
    final_data = []
    counter = 0
    for id in final_ranking:
        counter += 1
        final_data.append(id_2_row[id])
        # # have to do this rescoring for both post- and pre-process.
        # # new score = 1 - (pred_pos - 1)/NUM_ELEMENTS
        # final_data[-1][dataDescription.score_attribute] = 1.0 - float(counter-1)/length
        
    final_data = np.asarray(final_data)
    print(final_data.shape)
    final_data = pd.DataFrame(data=final_data, columns=new_header)

    
    
    # bring file into expected format for evaluation, if used for post-processing
    final_data.rename(columns = {'prot_attr':dataDescription.protected_group}, inplace = True)

    # write the data back    
    final_data.to_csv(dataDescription.result_path+'_delta='+str(p_deviation)+'.txt', sep=',', index=False, header=True)


def rerank_fair(dataDescription, dataset, p_deviation=0.0, iteration=None, rev=False):
    temp_header = dataDescription.header.copy()
    data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
    # data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
    data['uuid'] = 'empty'
    
    
    
    
    # for German credit, COMPAS and other such datasests, add query_id as all 1s.
    if 'query_id' not in data.columns.values:
        data['query_id'] = np.ones(len(data))
        dataDescription.header = np.append(dataDescription.header, 'query_id')
    

    if dataset == 'german':
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    if dataset == 'biased_normal':
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    
    if dataset == 'compas':
        if rev:
            data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        else:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
            data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    
    data['doc_id'] = data.index+1
    
    dataDescription.header = np.append(dataDescription.header, 'doc_id')

    
    np_data = np.array(data)
    # re-rank with fair for every query
    query = 1.0
    print("Rerank with FA*IR")
    data_query = data.query("query_id==" + str(query))
    data_query, protected, nonProtected = create(data_query, dataDescription)
    # protected attribute value is always 1
    # p = (len(data_query.query(dataDescription.protected_attribute + "==1")) / len(data_query) - p_deviation)
    
    p = float(len(data_query.query(dataDescription.protected_group + "==1")) / len(data_query) )
    p += p_deviation
    
    
    p = max(p, 0.01)
    print("lower bound for the protected group elements: ", p)
    fairRanking, _ = fair.fairRanking(data_query.shape[0], protected, nonProtected, p, dataDescription.alpha)
    
    # fairRanking = setNewQualifications(fairRanking)

    reranked_features = pd.DataFrame(columns=data_query[data_query.uuid == fairRanking[0].uuid].columns) 
    for candidate in fairRanking:
        # row = data_query.loc[data_query[data_query.uuid == candidate.uuid]
        row = data_query.loc[data_query['uuid'] == candidate.uuid]#.to_numpy()[0]
        reranked_features = reranked_features.append(row)
    
    
    # bring file into expected format for evaluation, if used for post-processing
    reranked_features = reranked_features.drop(columns=['uuid'])
    
    # write the data back    
    reranked_features.to_csv(dataDescription.result_path+'_delta='+str(p_deviation)+'.txt', sep=',', index=False, header=True)
    
    dataDescription.header = temp_header


def rerank_celis(dataDescription, dataset, p_deviation=0.0, iteration=None, rev=False, k=100):
    if 'german' in dataset:
        # dataDescription.header  ----   DurationMonth,CreditAmount,score,age25
        data = pd.read_csv(dataDescription.orig_data_path)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print(f"German Credit p value is: {p}, delta: {p_deviation}")
        
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        # for German credit, COMPAS and other such datasests, add query_id as all 1s.
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        
        # sort the training data based on true scores
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        
        # add 'doc_id' here itself
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')
        true_scores = data['score']
        
    elif 'biased_normal' in dataset:
        # dataDescription.header  ----   DurationMonth,CreditAmount,score,age25
        data = pd.read_csv(dataDescription.orig_data_path)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print("Biased Normal Synthetic p value is: ",p)
        
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        # for German credit, COMPAS and other such datasests, add query_id as all 1s.
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        
        # sort the training data based on true scores
        if rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        
        # add 'doc_id' here itself
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')
        true_scores = data['score']
        
    
    
    elif 'compas' in dataset:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
        p = []
        NUM_GROUPS = len(pd.unique(data[dataDescription.protected_group]))
        for i in range(NUM_GROUPS):
            proportion = float(len(data.query(dataDescription.protected_group + "=="+str(i))) / len(data)) 
            p.append(proportion)
        print("COMPAS p value is: ",p)
        
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        data['query_id'] = np.ones(len(data))
        new_header = np.append(dataDescription.header, 'query_id')
        if not rev:
            data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    
        data['doc_id'] = data.index+1
        new_header = np.append(new_header, 'doc_id')

        true_scores = data['Recidivism_rawscore']

    
    np_data = np.array(data)
    try:
        true_ranking = [f"id-{int(x)}" for x in data["doc_id"]] # 2nd column is ranking
        
    except KeyError:
        true_ranking = [f"id-{int(x+1)}" for x in range(len(data))]
    
    id_2_group = {}
    id_2_row = {}
    for idx, (id, group) in enumerate(zip(true_ranking, data["prot_attr"])):
        id_2_group[id] = int(group)
        id_2_row[id] = np_data[idx]
    
    # Now run the algo
    
    ##### GFair and Underranking version

    
    if rev:
        exp_flag = 2
    elif NUM_GROUPS > 2:
        exp_flag = 3
    else:
        exp_flag = 1
    solver = ConstrainedDP(true_ranking, true_scores, id_2_group, p=NUM_GROUPS, proportions=p, p_deviation=p_deviation, flag=exp_flag, k=k)
    final_ranking = solver.run_DP()
    
    # Contruct the dataframe back for dumping purposes
    length = len(final_ranking)
    final_data = []
    counter = 0
    for id in final_ranking:
        counter += 1
        final_data.append(id_2_row[id])
        
    final_data = np.asarray(final_data)
    final_data = pd.DataFrame(data=final_data, columns=new_header)
    
    
    
    # bring file into expected format for evaluation, if used for post-processing
    final_data.rename(columns = {'prot_attr':dataDescription.protected_group}, inplace = True)

    # write the data back    
    final_data.to_csv(dataDescription.result_path+'_delta='+str(p_deviation)+'.txt', sep=',', index=False, header=True)


def create(data, dataDescription):
    protected = []
    nonProtected = []

    for row in data.itertuples():
        # change to different index in row[.] to access other columns from csv file
        if row[data.columns.get_loc(dataDescription.protected_group) + 1] == 0.:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], [])
            nonProtected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid
            
        else:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], dataDescription.protected_group)
            protected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid
            
    # sort candidates by judgment
    protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
    nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)

    return data, protected, nonProtected


def setNewQualifications(fairRanking):
    qualification = len(fairRanking)
    for candidate in fairRanking:
        candidate.qualification = qualification
        qualification -= 1
    return fairRanking