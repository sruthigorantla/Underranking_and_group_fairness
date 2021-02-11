import pandas as pd
import numpy as np
# import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math
import os
from fileinput import filename


class Postprocessing_Evaluator():
    
    def __init__(self, dataset, resultDir, binSize, protAttr, topk, consecutive, start, rev):
        self.__trainingDir = '../data/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__dataset = dataset
        self.__k_for_evaluation = start + consecutive - 1
        self.rev = rev
        self.__block = consecutive
        if self.__block is not None:
            self.BLOCK_SIZE = consecutive
        if 'german' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['DurationMonth','CreditAmount','score',self.__prot_attr_name,'query_id','doc_id']

        elif 'biased_normal' in dataset:
            self.__prot_attr_name = 'prot_attr'
            self.__columnNames = ['score',self.__prot_attr_name,'query_id','doc_id']
              
        elif 'compas' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['priors_count','Violence_rawscore','Recidivism_rawscore',self.__prot_attr_name,'query_id','doc_id']

        else:
            self.__prot_attr_name = self.__dataset.split('-')[1]
            if self.__prot_attr_name == 'gender':
                self.__columnNames = ["query_id", "hombre", 'psu_mat', 'psu_len' ,'psu_cie', 'nem' ,'score','doc_id']
            else:
                self.__columnNames = ["query_id", "highschool_type", 'psu_mat', 'psu_len' ,'psu_cie', 'nem' ,'score','doc_id']
        self.__experimentNamesAndFiles = {}
        self.__results = {}
        self.axes_dict = {}
        self.figures = {}


    def evaluate(self):
        
        #### choose the directory where reranked results are stored
        if 'german' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'GermanCredit/'
        elif 'biased_normal' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'BiasedNormalSynthetic/'
        elif 'compas' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'COMPAS/'
        else:
            raise ValueError("Choose dataset from (enginering/compas/german)")
        
        
        #### create empty figures for plots
        # mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 4, 'lines.markersize': 40, 'font.family':'CMU Serif'})
        mpl.rcParams.update({'font.size': 5, 'lines.linewidth': 4, 'font.family':'DejaVu Sans'})
        plt.rcParams["axes.grid"] = False
        plt.rcParams['axes.linewidth'] = 1.25
        mpl.rcParams['axes.edgecolor'] = 'k'
        mpl.rcParams["legend.handlelength"] = 6.0
        self.figures['underranking_ndcg'] = plt.figure(figsize=(15,10))
        self.axes_dict['underranking'] = plt.axes()
        
        self.figures['representation_underranking'] = plt.figure(figsize=(15,10))
        self.axes_dict['representation'] = plt.axes()

        self.figures['rep'] = plt.figure(figsize=(15,10))
        self.axes_dict['rep'] = plt.axes()

        # self.figures['representation_fnr'] = plt.figure(figsize=(15,10))
        # self.axes_dict['representation'] = plt.axes()
        

        #### plot variables
        if self.rev:
            EXPERIMENT_NAMES = [ 'CELIS','ALG']
        else:
            EXPERIMENT_NAMES = [ 'FAIR','CELIS','ALG']
        METRIC_NAMES = ['underranking', 'ndcg', 'mfnr', 'representation', 'rep']

        colormap = {'ALG': 'black', 'FAIR': 'darkorange', 'CELIS': 'deepskyblue' } 

        linemap = {'underranking': 'solid', 'ndcg': 'dotted', 'representation': 'dashdot', 'rep': 'dashdot','fnr': 'dashed'}
        
        # markermap = {'ALG': 6, 'FAIR': 7, 'CELIS': '.'}
        markermap = {'ALG': '.', 'FAIR':'s', 'CELIS': '.'}

        markersizemap = {'ALG': 25, 'FAIR': 30, 'CELIS': 55}
        # markersizemap = {'ALG': 5, 'FAIR': 7, 'CELIS': 15}

        markerfillstyle = {'ALG': 'full', 'FAIR': 'none', 'CELIS': 'none'}

        lns1 = []
        lns2 = []
        lns3 = []
        for experiment in EXPERIMENT_NAMES:
            print(experiment)
            #### get the predictions for the experiment with all the deltas
            self.__predictions, self.__groundtruth = self.__prepareData(self.__trainingDir, experiment, self.__prot_attr_name)  
            #### calculate underranking and ndcg
            self.__results['underranking']  = self.__underranking()
            # print(f"Underranking: {self.__results['underranking']}")
            self.__results['ndcg'] = self.__ndcg()
            # print(f"NDCG: {self.__results['ndcg']}")
            self.__results['representation'], self.__true_rep_k, self.__true_rep_all = self.__representation()
            # print(f"Representation: {self.__results['representation']}")
            
            ### TODO implement FNR

            self.__results['fnr'] = self.__fnr()
            # print(f"FNR: {self.__results['fnr']}\n")

            lns1 = self.__plot_with_delta_for_x('underranking', 'ndcg', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, lns1)
            lns2 = self.__plot_with_delta_for_x('representation', 'underranking', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, lns2)
            lns3 = self.__plot_only_rep('rep', experiment, colormap, linemap, markermap, markersizemap, markerfillstyle, lns3)
        
        

    def __prepareData(self, pathsToFold, experiment, prot_attr_name):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        pred_files = list()
        predictedScores = {}
        for filename in os.listdir(self.__trainingDir):
            if self.rev:
                if ("rev" in filename) and (experiment in filename) and (prot_attr_name in filename):
                    delta = float((filename.split('=')[1]).split('.txt')[0])
                    predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)
            else:
                if ("rev" not in filename) and (experiment in filename) and (prot_attr_name in filename):
                    delta = float((filename.split('=')[1]).split('.txt')[0])
                    predictedScores[delta] = pd.read_csv(self.__trainingDir+'/'+filename, sep=",", header=0)
        
        if 'german' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'GermanCredit_'+prot_attr_name+'.csv', sep=",", header=0)
            if self.rev:
                groundtruth['score'] = groundtruth['score'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['score'], ascending=False)).reset_index(drop=True)
        elif 'biased_normal' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'BiasedNormalSynthetic_'+prot_attr_name+'.csv', sep=",", header=0)
            if self.rev:
                groundtruth['score'] = groundtruth['score'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['score'], ascending=False)).reset_index(drop=True)
        elif 'compas' in self.__dataset:
            groundtruth = pd.read_csv(self.__trainingDir+'/'+'ProPublica_'+prot_attr_name+'.csv', sep=",", header=0)
            if not self.rev:
                groundtruth['Recidivism_rawscore'] = groundtruth['Recidivism_rawscore'].apply(lambda val: 1-val)
            groundtruth = (groundtruth.sort_values(by=['Recidivism_rawscore'], ascending=False)).reset_index(drop=True)
        
        groundtruth['doc_id'] = np.arange(len(groundtruth))+1
        return predictedScores, groundtruth


    def __underranking(self):
        '''
        calculate underranking in top-k for all the deltas for the given experiment.
        underranking = max multiplicative displacement of an item.

        --- EXAMPLE BEGINS ---
        1,2,4,6,5,3 pred
        1,2,6,3,5,4 sorted indices
        1,2,3,4,5,6 true (i+1)

        underranking for k = 3 will be 6/3 = 2

        --- EXAMPLE ENDS ---

        '''
        underranking_results = {}
        for delta, preds in self.__predictions.items():
            if 'doc_id' in preds.columns.values:
                temp = preds['doc_id']
            else:
                temp = preds['rank']
            predicted_ranking = temp.reset_index(drop=True).to_numpy()
            
            new_pred_ranking = predicted_ranking.copy()
            for i in range(len(self.__groundtruth)):
                if i+1 not in new_pred_ranking:
                    new_pred_ranking = np.concatenate((new_pred_ranking, [i+1]))
            
            sorted_indices = np.argsort(new_pred_ranking)+1
            ans = 1
            
            for i in range(self.__k_for_evaluation):
                if sorted_indices[i]/float(i+1) > ans:
                    ans = sorted_indices[i]/float(i+1)
            
           
            underranking_results[delta] = ans
            
            
            
        return underranking_results
        
    def __ndcg(self):
        '''
        calculate ndcg in top-k for all the deltas for the given experiment
        
        ndcg@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}
        
        '''
        
        ### calculate maximum ndcg possible
        
        idcg = 0
        data = self.__groundtruth
        for i in range(self.__k_for_evaluation):
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                try:
                    score = data.loc[data['doc_id'] == i+1, 'score'].iloc[0]
                except IndexError:
                    score = 0
            elif 'compas' in self.__dataset:
                score = data.loc[data['doc_id'] == i+1, 'Recidivism_rawscore'].iloc[0]
            idcg += (2**score-1)/(np.log(i+2))
            
        ndcg_results = {}
        ### calculate dcg in the top k predicted ranks
        for delta, preds in self.__predictions.items():
            preds_k = preds.head(self.__k_for_evaluation)
            if 'german' in self.__dataset or 'biased_normal' in self.__dataset:
                scores = preds_k['score'].reset_index(drop=True).to_numpy()
            elif 'compas' in self.__dataset:
                scores = preds_k['Recidivism_rawscore'].reset_index(drop=True).to_numpy()
            dcg = 0
            for i in range(self.__k_for_evaluation):
                dcg += (2**scores[i]-1)/(np.log(i+2))
            ndcg_results[delta] = dcg/idcg
        
        
        
        return ndcg_results   
    
    def __representation(self):
        '''
        calculate representation of the protected group in top-k for all the deltas for the given experiment
        
        representationProt@k = #protected@k/ k

        '''

        #### calculate groundtruth representation
        if self.__block is not None:
            true_data = self.__groundtruth
            true_data_k = self.__groundtruth.iloc[self.__block - self.BLOCK_SIZE : self.__block]
            true_rep_k = float(len(true_data_k.loc[true_data_k[self.__prot_attr_name] == 1.0]))/self.BLOCK_SIZE
            true_rep_all = float(len(true_data.loc[true_data[self.__prot_attr_name] == 1.0]))/len(true_data)

            representation_results = {}
            for delta, preds in self.__predictions.items():
                preds_k = preds.iloc[self.__block - self.BLOCK_SIZE : self.__block]
                s = len(preds_k.loc[preds_k[self.__prot_attr_name] == 1.0])
                representation_results[delta] = float(s)/self.BLOCK_SIZE
                
            return representation_results, true_rep_k, true_rep_all
        elif self.__k_for_evaluation is not None:
            true_data = self.__groundtruth
            true_data_k = self.__groundtruth.head(self.__k_for_evaluation)
            true_rep_k = float(len(true_data_k.loc[true_data_k[self.__prot_attr_name] == 1.0]))/self.__k_for_evaluation
            true_rep_all = float(len(true_data.loc[true_data[self.__prot_attr_name] == 1.0]))/len(true_data)

            representation_results = {}
            for delta, preds in self.__predictions.items():
                preds_k = preds.head(self.__k_for_evaluation)
                s = len(preds_k.loc[preds_k[self.__prot_attr_name] == 1.0])
                representation_results[delta] = float(s)/self.__k_for_evaluation
                
            return representation_results, true_rep_k, true_rep_all


    def __fnr(self):
        '''
        calculate difference of fnr of the non-protected and the protected group in top-k for all the deltas for the given experiment
        
        fnr@k = 

        '''
        fnr_results = {}
        for delta, preds in self.__predictions.items():
            if 'doc_id' in preds.columns.values:
                temp = preds['doc_id']
            else:
                temp = preds['rank']
            predicted_ranking = temp.reset_index(drop=True).to_numpy()
            s = 0
            for i in range(self.__k_for_evaluation):
                s += (self.__groundtruth.loc[self.__groundtruth['doc_id'] == predicted_ranking[i], self.__prot_attr_name].iloc[0]) 
            fnr_results[delta] = float(s)/self.__k_for_evaluation
        return fnr_results   

    
    def __plot_with_delta_for_x(self, metric1, metric2, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, lns=None):
        
        p = self.__true_rep_all
        p_k = self.__true_rep_k        
                
        all_deltas = list(self.__results[metric1].keys())
        deltas = []
        for delta in all_deltas:
            if delta + p > 0:
                deltas.append(delta)
        deltas = np.sort(deltas)
        result1 = [self.__results[metric1][i] for i in deltas]
        result2 = [self.__results[metric2][i] for i in deltas]

        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k\' = $'+str(self.__k_for_evaluation)
        if self.__prot_attr_name == 'sex':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=30)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=30)
        self.axes_dict[metric1].set_xlabel('$\delta$', fontsize=30)
        self.axes_dict[metric1].set_ylabel(metric1, fontsize=30)
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=30)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=30)
        self.axes_dict[metric1].set_facecolor("white")
        
        
        # self.axes_dict[metric1].spines['bottom'].set_color('black')
        # self.axes_dict[metric1].spines['bottom'].set_visible(True)
        # self.axes_dict[metric1].spines['left'].set_color('black')
        # self.axes_dict[metric1].spines['left'].set_visible(True)
        # self.axes_dict[metric1].spines['right'].set_color('black')
        # self.axes_dict[metric1].spines['top'].set_color('black')
        
        # self.axes_dict[metric1].grid(color='silver', linestyle='dotted', linewidth=0.7)
        # self.axes_dict[metric1].legend(loc="lower right", prop={'size': 20}, facecolor='white', framealpha=1)

        ax2 = self.axes_dict[metric1].twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(metric2, fontsize=27)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', which='minor', labelsize=27)
        ax2.tick_params(axis='y', which='major', labelsize=27)
        if self.rev:
            print(p, p_k)
            if metric2 == 'ndcg':
                ax2.set_ylim(0.90, 1.01)#(0.75,1.1)
            if metric1 == 'representation':
                self.axes_dict[metric1].set_ylim(p-0.2, p_k + 0.5)    
        else:
            if metric2 == 'ndcg':
                ax2.set_ylim(0.90, 1.01)#(0.75,1.1)
            elif metric2 == 'underranking':
                ax2.set_ylim(0.5,4.5)#(0.8,5)    
            if metric1 == 'underranking':
                self.axes_dict[metric1].set_ylim(0.5,4.5)#(0.8,5)    
            elif metric1 == 'representation':
                self.axes_dict[metric1].set_ylim(p_k - 0.1,p+0.3)  
        ##### , fillstyle='none',markersize=3
        markerwidthval = 4
        if self.rev:
            max_ind = len(deltas)
        else:
            max_ind = len(deltas[:-2])
        if "FAIR" in experiment:
            name = 'FA*IR'
            lns1 = self.axes_dict[metric1].plot((deltas-0.05)[1:-1], result1[1:-1], color=colormap[experiment], label=name+' ('+metric1+')', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            lns2 = ax2.plot((deltas-0.05)[1:-1], result2[1:-1], color=colormap[experiment], label=name+' ('+metric2+')', marker=markermap[experiment], linestyle=linemap[metric2],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        else:
            name = experiment
            if "ALG" in experiment or 'CELIS' in experiment:
                lns1 = self.axes_dict[metric1].plot(deltas[:max_ind], result1[:max_ind], color=colormap[experiment], label=name+' ('+metric1+')', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            else:
                lns1 = self.axes_dict[metric1].plot(deltas[:max_ind], result1[:max_ind], color=colormap[experiment], label=name+' ('+metric1+')', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            # ax2.grid(color='silver', linestyle='dotted', linewidth=0)
            if "ALG" in experiment or 'CELIS' in experiment:
                lns2 = ax2.plot(deltas[:max_ind], result2[:max_ind], color=colormap[experiment], label=name+' ('+metric2+')', marker=markermap[experiment], linestyle=linemap[metric2],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            else:
                lns2 = ax2.plot(deltas[:max_ind], result2[:max_ind], color=colormap[experiment], label=name+' ('+metric2+')', marker=markermap[experiment], linestyle=linemap[metric2],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            if metric1 == 'representation':
                self.axes_dict[metric1].plot(deltas[:max_ind], [p]*len(deltas[:max_ind]),  label='y = ${p^*}$', color='limegreen', linestyle='dashed', linewidth=6.0)
                self.axes_dict[metric1].plot(deltas[:max_ind], [p_k]*len(deltas[:max_ind]),  label='y = ${\hat{p}}$', color='crimson', linestyle='dashed', linewidth=6.0)

        
        # added these three lines
        every_nth = 2
        # for n, label in enumerate(self.axes_dict[metric1].xaxis.get_ticklabels()):
        #     if n % every_nth != 0:
        #         label.set_visible(False)
        for n, label in enumerate(self.axes_dict[metric1].yaxis.get_ticklabels()):
            if (n-1) % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax2.yaxis.get_ticklabels()):
            if (n-1) % every_nth != 0:
                label.set_visible(False)
        
        lns += lns1
        lns += lns2
        
        labs = [l.get_label() for l in lns]
        
        # self.axes_dict[metric1].legend(lns, labs, prop={'size': 20}, facecolor='white', loc='upper center',ncol=2, framealpha=1, bbox_to_anchor=(0.49, 1.1))
        # self.axes_dict[metric1].legend(lns, labs, prop={'size': 7.5}, facecolor='white', loc='upper center',ncol=3, framealpha=1)

        self.figures[metric1+'_'+metric2].savefig(self.__resultDir + metric1+'_'+metric2+'_' + self.__dataset + '.png', dpi=300)
        self.figures[metric1+'_'+metric2].savefig(self.__resultDir + metric1+'_'+metric2+'_' + self.__dataset + '.pdf')

        return lns



    def __plot_only_rep(self, metric1, experiment,  colormap, linemap, markermap, markersizemap, markerfillstyle, lns=None):
        
        p = self.__true_rep_all
        p_k = self.__true_rep_k        
                
        all_deltas = list(self.__results['representation'].keys())
        deltas = []
        for delta in all_deltas:
            if delta + p > 0:
                deltas.append(delta)
        deltas = np.sort(deltas)
        result1 = [self.__results['representation'][i] for i in deltas]
        
        if self.__block is not None:
            k_str = '$k\' = $'+str(self.__block - self.BLOCK_SIZE+1)+' to '+str(self.__block)
        else:
            k_str = '$k\' = $'+str(self.__k_for_evaluation)
        if self.__prot_attr_name == 'sex':
            self.axes_dict[metric1].set_title('Protected group = female, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'age35':
            self.axes_dict[metric1].set_title('Protected group = ${age < 35}$, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'age25':
            self.axes_dict[metric1].set_title('Protected group = ${age < 25}$, '+k_str, fontsize=30)
        elif self.__prot_attr_name == 'race':
            self.axes_dict[metric1].set_title('Protected group = African American, '+k_str, fontsize=30)
        else:
            self.axes_dict[metric1].set_title('Protected group = '+self.__prot_attr_name+', '+k_str, fontsize=30)
        self.axes_dict[metric1].set_xlabel('$\delta$', fontsize=30)
        self.axes_dict[metric1].set_ylabel('representation', fontsize=30)
        self.axes_dict[metric1].tick_params(axis='both', which='major', labelsize=30)
        self.axes_dict[metric1].tick_params(axis='both', which='minor', labelsize=30)
        self.axes_dict[metric1].set_facecolor("white")
        
        
        
        if self.rev:
            self.axes_dict[metric1].set_ylim(p-0.2, p_k + 0.5)    
        else:
            self.axes_dict[metric1].set_ylim(p_k - 0.1,p+0.3)  
        ##### , fillstyle='none',markersize=3
        markerwidthval = 4
        if "FAIR" in experiment:
            name = 'FA*IR'
            lns1 = self.axes_dict[metric1].plot((deltas-0.05)[1:-1], result1[1:-1], color=colormap[experiment], label=name+' (representation)', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
        else:
            name = experiment
            if "ALG" in experiment or 'CELIS' in experiment:
                lns1 = self.axes_dict[metric1].plot(deltas[:-2], result1[:-2], color=colormap[experiment], label=name+' (representation)', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            else:
                lns1 = self.axes_dict[metric1].plot(deltas[:-2], result1[:-2], color=colormap[experiment], label=name+' (representation)', marker=markermap[experiment], linestyle=linemap[metric1],markersize=markersizemap[experiment], fillstyle=markerfillstyle[experiment], markeredgewidth=markerwidthval)
            if metric1 == 'rep':
                self.axes_dict[metric1].plot(deltas[:-2], [p]*len(deltas[:-2]),  label='y = ${p^*}$', color='limegreen', linestyle='dashed', linewidth=6.0)
                self.axes_dict[metric1].plot(deltas[:-2], [p_k]*len(deltas[:-2]),  label='y = ${\hat{p}}$', color='crimson', linestyle='dashed', linewidth=6.0)

        
        # added these three lines
        # every_nth = 2
        # for n, label in enumerate(self.axes_dict[metric1].yaxis.get_ticklabels()):
        #     if (n-1) % every_nth != 0:
        #         label.set_visible(False)
        
        
        lns += lns1
        
        labs = [l.get_label() for l in lns]
        
        # self.axes_dict[metric1].legend(lns, labs, prop={'size': 20}, facecolor='white', loc='upper center',ncol=2, framealpha=1, bbox_to_anchor=(0.49, 1.1))
        # self.axes_dict[metric1].legend(lns, labs, prop={'size': 7.5}, facecolor='white', loc='upper center',ncol=2, framealpha=1)

        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.png', dpi=300)
        self.figures[metric1].savefig(self.__resultDir + metric1+'_' + self.__dataset + '.pdf')

        return lns