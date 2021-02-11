import random
import numpy as np
from collections import defaultdict
from itertools import product
import math
import sys

class ConstrainedDP:

	def __init__(self, true_ranking, true_scores, id_2_group, p, proportions, p_deviation, flag, k):
		self.M = len(true_ranking)
		self.N = k
		self.p = p # for our case, p = q
		self.proportions = proportions
		self.flag = flag
		## flag: 1 (lower bounds for 2 groups), 2 (upper bounds for 2 groups), 3 (multigroups)
		# Generate utility matrix (if required)
		# Note that W is created as 1-based indexing, i.e. W[1,1] = utility of having item 1 in rank 1. Similarly, W[1,0] or W[0,1] or W[0,0] is not defined 
		self.W = self.generate_W(true_scores)

		# Generate lower and uppper fairness bounds for each prefix length and for each property
		# The bounds are stored in a dictionary where key = prefix length and value = vector of bound values of size p (i.e. for each property one value)
		self.LB, self.UB = self.get_bounds(p_deviation)

		self.generate_data(true_ranking, id_2_group)

		# Initialize the DP table
		self.DP = self.initialize_dp()


	def generate_W(self, true_scores):
		# This function generates a dummy utility matrix
		# Ideally, this should be loaded from a dataset

		# Ensure that W is created as 1-based indexing
		W = np.zeros((self.M+1, self.N+1))

		for idx in range(1,self.M+1):
			val = true_scores[idx-1]
			for idy in range(1,self.N+1):
				W[idx, idy] = val/np.log(idy+1)				

		# Check properties of W
		# self.check_W(W)
		return W

	def check_W(self, W):
		# Checks if entries in utility matrix W is monotonic and satisfying Monge
		for i1 in range(1,W.shape[0]):
			for i2 in range(i1+1, W.shape[0]):
				for j1 in range(1,W.shape[1]):
					for j2 in range(j1+1, W.shape[1]):
						assert W[i1,j1] >= W[i2, j1], "Monotonicity not satisfied"
						assert W[i1,j1] >= W[i1, j2], "Monotonicity not satisfied"
						assert (W[i1,j1] + W[i2, j2]) >= (W[i1,j2] + W[i2, j1]), "Monge not satisfied"
		print("W satisfies the monotonic properties")

		assert sum(W[:,0]) == 0, "Ensure that W is created as 1-based indexing"
		assert sum(W[0,:]) == 0, "Ensure that W is created as 1-based indexing"

	def get_bounds(self, delta):
		# print(delta)
		# # Generating L_k and U_k vectors
		L_k, U_k = {}, {}
		# ALPHAS = [1, 1]
		# BETAS = [0, self.proportions[1] + delta]
		# # B = math.floor(max((1+self.p/(1.0-sum(BETAS))), (1+self.p/(sum(ALPHAS) - 1.0)), (1+2/(ALPHAS[1] - BETAS[1]))) )
		# B = 20
		for k in range(self.N):
			if self.flag == 1:
				## lower bounds
				upper = [k+1, k+1]
				lower = [0, math.ceil((self.proportions[1]+delta)*(k+1))]
			elif self.flag == 2:
				## for upper bounds, for reverse ranking
				upper = [k+1, math.floor((self.proportions[1]+delta)*(k+1))]
				lower = [0, 0]
			elif self.flag == 3:
				upper = []
				lower = []
				for j in range(self.p):
					upper.append(math.ceil((self.proportions[j]+delta)*(k+1)))
					lower.append(math.floor((self.proportions[j]-delta)*(k+1)))
					
			L_k[k+1] = lower
			U_k[k+1] = upper
		# There should be lower and upper bound vectors for each prefix k
		
		assert len(L_k) == self.N 
		assert len(U_k) == self.N 
		return L_k, U_k

	def generate_data(self, true_ranking, id_2_group):

		
		# Assign a random type for each item 
		self.types = types = []
		for idx in range(self.M):
			# Sample a random property for i^th item
			# c_type = random.randint(0, self.p-1)
			# one_hot = [0]*self.p
			c_type = id_2_group['id-'+str(idx+1)]

			one_hot = [0]*self.p
			one_hot[c_type]=1 # Asign one-hot vector for the sampled property to the item
			types.append(one_hot)
		
		# get unique types (v_i's from the paper)
		self.unique_types = unique_types = list(set([tuple(c_list) for c_list in types])) # Dimension of unique_types = q
		self.q = len(unique_types) 

		# Get ranked list of items for each type (Q_l from the paper)
		self.item_map = item_map = defaultdict(list) # key : unique type, value : list of items with the type in sorted order
		for item_id, type in enumerate(types):
			item_map[tuple(type)].append(item_id+1) # Items are 1-based ids and not 0-based
		
	def initialize_dp(self):

		dimensions = [self.N+1]*self.q
		DP = np.zeros(dimensions, dtype=np.ubyte)

		# Fill all entries by infinite
		DP.fill(-np.inf)

		# set DP[0,0,...,0] = 0
		indices = tuple(0 for _ in range(self.q))
		DP[indices] = 0
		
		return DP

	def fairness_check(self, candidate):
		# Checks whether a given 
		assert len(candidate) == len(self.unique_types) # both should be of length q
		prefix_len = sum(candidate)


		v = np.array([0.0]*self.p)
		for s_i, v_i in zip(candidate,self.unique_types):
			v += s_i*np.array(v_i)

		# print(v, self.LB[prefix_len], self.UB[prefix_len])

		# Fairness checks for current prefix_len
		if False in (self.LB[prefix_len]<=v):
			return False
		
		if False in (self.UB[prefix_len]>=v):
			return False

		return True

	def run_DP(self):
		# Run the DP algorithm
		# TODO: The tuple looping could be done more smartly

		###########################
		# generate all s tuples
		candidate_s = []
		for type in self.unique_types:
			# get the number of items each type has
			num_items = len(self.item_map[type])
			candidate_s.append([idx for idx in range(num_items+1)])
		
		# Generate cross product of candidate_s
		tuples = []
		for items in product(*candidate_s):
			tuples.append(items)
		###########################


		solution = {}
		
		# is_infeasible = False
		# Loop over increasing k ranks
		for k in range(1, self.N+1):
			sys.stdout.flush()
			
			# Iterate over all valid tuples with sum = k
			for candidate in tuples:

				# check if current candidate sums to k
				if sum(candidate) != k:
					continue


				# First need to ensure the current candidate is satisfying fairness bounds
				if (self.fairness_check(candidate)):
				
					# Check for all types
					for l in range(self.q):
						prev_candidate = list(candidate)
						prev_candidate[l] -= 1
						prev_candidate = tuple(prev_candidate)
						
						if prev_candidate in tuples:
							# get item id = s_l^{th} item of l^{th} type
							s_l = list(candidate)[l]
							assert s_l > 0
							item_to_consider = self.item_map[self.unique_types[l]][s_l-1]
							# Tracking which item is placed at k^{th} rank and updating the DP table:
							if (self.DP[prev_candidate] + self.W[item_to_consider,k]) >= self.DP[candidate]:
								solution[candidate] = (item_to_consider, prev_candidate)
								self.DP[candidate] = self.DP[prev_candidate] + self.W[item_to_consider,k]
								# if k == 100 or k == 200:
								# 	print(k, self.DP[candidate], candidate)
		# Do backtracking


		# First iterate overall all valid candidates of rank N and see which one has highest DP value. 
		# Then bactrack from it
		max_N = -np.inf
		candidate_N = None
		for candidate in tuples:
			if sum(candidate) == self.N:
				if (self.DP[candidate] > -np.inf) and (self.DP[candidate] > max_N):
					max_N = self.DP[candidate]
					candidate_N = candidate

		if candidate_N is None:
			print("Infeasible Solution")
		else:
			# Start backtracking
			final_ranking = [solution[candidate_N][0]]
			back_pointer_candidate = solution[candidate_N][1]
			while sum(back_pointer_candidate)>0:
				final_ranking.append(solution[back_pointer_candidate][0])
				back_pointer_candidate = solution[back_pointer_candidate][1]
		
		assert len(final_ranking) == self.N, "Something wrong"

		
		final_indices = []
		for i in final_ranking[::-1]:
			final_indices.append("id-"+str(i))
		return final_indices



if __name__ == "__main__":


	M = 50 # No. of items
	N = 10 # No. of ranks
	p = 2  # No. of properties (e.g. p = 3 might mean {male, female, others})
	# Note q ,i.e. no. of uniquw types get automatically calculated in the class. Also, in this code, q turns out to be = p

	solver = ConstrainedDP(M=M, N=N, p=p)
	final_rank, is_infeasible = solver.run_DP()

	if not is_infeasible:
		print(final_rank)
	else:
		print("Infeasible")