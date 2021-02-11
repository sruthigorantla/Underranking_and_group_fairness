import processingWithFair.rerank_for_fairness as rerank
from processingWithFair.DatasetDescription import DatasetDescription
import sys

class Postprocessing():

    def __init__(self, args):
        self.dataset = args.postprocess[0]
        self.method = args.postprocess[1]
        self.k = args.k
        self.multi_group = args.multi_group
        self.rev_flag = args.rev_flag
        # self.deltas = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
        self.deltas = [-0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        # self.deltas = [0.0, 0.05, 0.10, 0.15, 0.20]
        # self.deltas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
        # self.deltas = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0]
        # self.deltas = [ 0.0]

    def postprocess(self):

        if self.dataset == "german" and not self.multi_group:

            """
            German Credit dataset - age 25

            """
            print("Start reranking of German Credit - Age 25")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age25"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age25']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age25.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age25_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age25_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)

            """
            German Credit dataset - age 35

            """
            print("Start reranking of German Credit - Age 35")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age35"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age35']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age35.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age35_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age35_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)

        elif self.dataset == "german" and self.multi_group:
            """
            German Credit dataset - age

            """
            print("Start reranking of German Credit - Age")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)

            """
            German Credit dataset - age_gender

            """
            print("Start reranking of German Credit - Age and Gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age_gender"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age_gender']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age_gender.csv"
            if self.rev_flag:
                resultFile = "../data/GermanCredit/GermanCredit_age_gender_rev_" + self.method 
            else:
                resultFile = "../data/GermanCredit/GermanCredit_age_gender_" + self.method 
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(GermanCreditData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)

            
        
        
        elif self.dataset == 'compas':

            """
            COMPAS propublica dataset - race

            """
            print("Start reranking of COMPAS propublica - Race")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "race"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','race']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_race.csv"
            if self.rev_flag:
                resultFile = "../data/COMPAS/ProPublica_race_rev_" + self.method 
            else:
                resultFile = "../data/COMPAS/ProPublica_race_" + self.method 
            CompasData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)

            """
            COMPAS propublica dataset - gender

            """
            print("Start reranking of COMPAS propublica - gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "sex"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','sex']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_sex.csv"
            if self.rev_flag:
                resultFile = "../data/COMPAS/ProPublica_sex_rev_" + self.method 
            else:
                resultFile = "../data/COMPAS/ProPublica_sex_" + self.method 
            CompasData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
            for i in range(len(self.deltas)):
                sys.stdout.flush()
                if "ALG" in self.method:
                    rerank.rerank_alg(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i, k=self.k,  rev=self.rev_flag)
                elif "FAIR" in self.method:
                    rerank.rerank_fair(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag)
                elif "CELIS" in self.method:
                    rerank.rerank_celis(CompasData, self.dataset, p_deviation=self.deltas[i], iteration=i,  rev=self.rev_flag, k=self.k)
