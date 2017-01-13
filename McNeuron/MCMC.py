"""
Perturbaion class for doing MCMC on neuron
It gives a neuron and preforms iteration on the neuron and save it based on the Metropolis_Hasting methods

"""
import sys
import numpy as np
from numpy import linalg as LA
from McNeuron import Neuron
from McNeuron import Node
from McNeuron import visualize
import matplotlib.pyplot as plt
from numpy import mean, cov, dot, linalg, transpose
from __builtin__ import str
from copy import deepcopy
from scipy.stats import chi2
from scipy.stats import vonmises
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import scipy.special
#np.random.seed(500)

class MCMC(object):

    def __init__(self,
                neuron = None,
                measures_on_features = None,
                MCMCs_list = None,
                n_node = 10,
                initial_seg = 4,
                MCMC_prob = None,
                iterations = 100,
                mean_len_to_parent = 5,
                var_len_to_parent = 1,
                verbose = 0):
        """
        Default class for perturbation doing MCMC over a set of features

        Parameters:
        -----------
        neuron: :class: `Neuron`
            The starting neuron, by default it starts with a simple neuron with only one soma and one compartment.

        ## features: list
        ##    The list of all features to perform MCMC. The default is the list of features appears in paper ???
        ##    e.g. features = ['Volume', 'length', 'Sholl']

        measures_on_features: dict
            The probability distribution of each feature. Use can costumize it by given the emprical distribution of the dataset.
            The default use:
                Possion: for integer values (e.g. number of the branching points,...)
                Gamma: for positive and integer values (e.g. length of the neuron)
                Wrapped Normal distribution: for continous and limited values (e.g. angles,...)
                Normal distribution: for real and continous feature (e.g. 3D vector approximation of a neuron)

        MCMCs_list: list
            List of all proposals for MCMC. see do_MCMC function for details of all proposals

        MCMC_prob: `numpy array`
            The probability of choosing one of the operations

        iterations: integer
            Number of iterations.

        verbose: integer
            0: for saving only the last neuron.
            1: for saving the evolution neuron by saving all intermediae neurons.
            2: for saving the intermediate neuron in addition to feature diagram of them.
        """
        self.ratio_red_to_ext = 1.
        self.n_node = n_node
        self.initial_seg = initial_seg
        if(neuron == None): # make a simple neuron with one soma and one apical node
            self.neuron = self.initial_neuron(int(self.n_node/self.initial_seg),self.initial_seg)
        else:
            self.neuron = neuron

        if(measures_on_features == None):
            self.measure = {'mean' : {'Nnodes':2000, 'Nbranch':30,'Nendpoint':30,'local_angle':2.5,'global_angle':0.5,'slope':0  ,'diameter':.2,'length_to_parent':3,
                                      'distance_from_root':100, 'branch_angle':2.4, 'ratio_euclidian_neuronal':1.2},

                        'variance' : {'Nnodes':400, 'Nbranch':10,'Nendpoint':10,'local_angle':0.5,'global_angle':0.5,'slope':0.02,'diameter':0.2,'length_to_parent':.2,
                                      'distance_from_root':30 , 'branch_angle':.04, 'ratio_euclidian_neuronal':1}}
        else:
            self.measure = measures_on_features

        if(MCMCs_list == None):
            self.p_list = ['extension/reduction','extension/reduction end points','sliding', 'add/remove', 'location', 'location for important point', 'location toward end', 'diameter', 'diameter_toward', 'rotation for any node', 'rotation for branching' , 'sliding general', 'rescale toward end',  'sliding certain in distance','sliding for branching node', 'stretching vertical', 'stretching horizental', 'sinusidal']
        else:
            self.p_list = MCMCs_list

        if(MCMC_prob == None):
            self.p_prob  = np.array([.01,.09,.65,.25,.1])
        else:
            self.p_prob = MCMC_prob
        self.erf = 0.5*(3+scipy.special.erf(-mean_len_to_parent/np.sqrt(2)))
        self.mean_len_to_parent = mean_len_to_parent
        self.var_len_to_parent = var_len_to_parent
        self._consum_prob = np.array([sum(self.p_prob[0:i]) for i in range(len(self.p_prob))])
        self.ite = iterations
        self.verbose = verbose
        self.trace_MCMC = [0, self.neuron]
        self.cte_gauss = 1/np.sqrt(2*np.pi)
        self.kappa = 4.
        self.kappa_rotation = 50.
        self.n_chi = 30.0 # float number
        self.mu = .1
        self.mean_ratio_diameter = .1 # float number
        self.mean_loc = 1.0 # float number
        self.var = 1.0/np.sqrt(3)
        self.list_features = self.measure['mean'].keys()
        self.mean_measure = np.array([])
        self.variance_measure = np.array([])
        self.sliding_limit = 20*20
        self.rescale_value = 1
        self.horizental_stretch = .1
        self.vertical_stretch = .1
        self.location_toward_cte = .4
        self.location = .1
        self.location_important = .1
        self.sinusidal_hight = .01
        self.evo = []

        #self.trend = np.zeros([len(self.measure['mean']),self.ite])

    def fit(self):
        """
        The main function for MCMC. It starts with current neuron and in each iteration selects one of the pertubations form the probability distribution of MCMC_prob and does it on the neuron.
        """
        # if self.verbose == 1:
        #     print ('The list of all perturbation with its probability: \n')
        #     for perturb in range(len(self.p_list)):
        #         print('%s perturbation with probability of: %s \n' %(self.p_list[perturb], self.p_prob[perturb]))
        #p_current, error_vec_current ,error_vec_normal_current = self.minus_log_prob_neuron(self.neuron) # log probability of the current neuron
        p_current, error_vec_current ,error_vec_normal_current = self.kl_distance(self.neuron) # log probability of the current neuron
        acc = 0
        for i in range(self.ite):
            if(self.verbose ==1):
                #p_current, er , error_vec_normal_current = self.minus_log_prob_neuron(self.neuron)
                p_current, er , error_vec_normal_current = self.kl_distance(self.neuron)
                #print('feature of current is: \n %s' %(self.neuron.features)+ '\n')
                print('\n and its probability is: %s' %p_current)
            per = self.select_proposal() # MCMC index
            p_sym, details = self.do_MCMC(per)
            #p_proposal, error_vec_proposal, error_vec_normal_proposal = self.minus_log_prob_neuron(self.neuron)
            p_proposal, error_vec_proposal, error_vec_normal_proposal = self.kl_distance(self.neuron)
            if(self.verbose ==1):
                #print('feature of proposal is: \n %s' %(self.neuron.features))
                print('\n and its probability is: %s' %p_proposal)
            a = min(1, p_sym * np.exp(p_current - p_proposal)) # Metropolis choice, notice that the values are minus log probability
            B = self.accept_proposal(a) # the boolean of acceptance
            if(B):
                p_current = p_proposal
                error_vec_current = error_vec_proposal
                error_vec_normal_current = error_vec_normal_proposal
                self.trend[:,i] = error_vec_proposal
                self.trend_normal[:,i] = error_vec_normal_proposal
                acc = acc + 1
            else:
                self.undo_MCMC(per, details)
                self.trend[:,i] = error_vec_current
                self.trend_normal[:,i] = error_vec_normal_current
            if len(self.neuron.nodes_list) == self.neuron.n_soma:
                self.neuron = self.initial_neuron(int(self.n_node/self.initial_seg),self.initial_seg)
                #p_current, error_vec_current, error_vec_normal_current = self.minus_log_prob_neuron(self.neuron)
                p_current, error_vec_current, error_vec_normal_current = self.kl_distance(self.neuron)
            if(self.verbose ==1):
                print ('\n')
                print('Selected perturbation = ' + per)
                print('the p of acceptance was %s and it was %s that it`s been accepted.'%(a,B))
                print ('\n')
            if(np.remainder(i,100)==0):
                self.evo.append(deepcopy(self.neuron))
        self.neuron.set_nodes_values()
        print acc

    def set_ratio_red_to_ext(self,c):
        self.ratio_red_to_ext = c
        self.neuron.set_ratio_red_to_ext(c)

    def set_n_node(self,n):
        self.n_node = n
        self.neuron = self.initial_neuron(n)

    def set_verbose(self, n):
        self.verbose = n

    def set_initial_neuron(self,neuron):
        self.neuron = neuron

    def set_n_iteration(self, n):
        self.ite = n

    def initial_neuron(self,n, k = 1):
        L = []
        root = Node()
        root.r = .2
        root.type = 'soma'
        root.xyz = np.array([0,0,0],dtype = float)
        L.append(root)
        for i in range(3):
            soma = Node()
            soma.r = .2
            soma.type = 'soma'
            soma.xyz = np.array([0,0,0],dtype = float)
            L.append(soma)
            root.add_child(soma)
            soma.parent = root
        for j in range(k):
            for i in range(n):
                node = Node()
                node.r = .2
                node.type = 'apical'
                node.xyz = np.array([np.sin(j*2*np.pi/k)*10*(i+1),np.cos(j*2*np.pi/k)*10*(i+1),0.],dtype = float)#+0*np.random.rand(3)

                if i == 0:
                    root.add_child(node)
                    node.parent = root
                    L.append(node)
                else:
                    L[-1:][0].add_child(node)
                    node.parent = L[-1:][0]
                    L.append(node)
        neuron = Neuron(file_format = 'only list of nodes', input_file = L)
        return neuron

    def do_MCMC(self, per):
        if per == 'extension/reduction': # do extension/reduction
            p_sym, details = self.do_ext_red(self.neuron)
        if per == 'extension/reduction end points': # do extension/reduction end points
            p_sym, details = self.do_ext_red_end_points(self.neuron)
        if per == 'add/remove': # do add/remove
            p_sym, details = self.do_add_remove_node(self.neuron)

        if per == 'diameter': # do diameter
            p_sym, details = self.do_diameter(self.neuron)
        if per == 'diameter_toward': # do diameter_toward
            p_sym, details = self.do_diameter(self.neuron)

        if per == 'rotation for any node': # do general rotation
            p_sym, details = self.do_rotation_general(self.neuron, self.kappa_rotation)
        if per == 'rotation for branching': # do branch point rotation
            p_sym, details = self.do_rotation_branching(self.neuron, self.kappa_rotation)

        if per == 'location toward end': # do location toward end
            p_sym, details = self.do_location_toward_end_nodes(self.neuron)
        if per == 'location': # do location
            p_sym, details = self.do_location(self.neuron)
        if per == 'location for important point':
            p_sym, details = self.do_location_important(self.neuron)

        if per == 'rescale toward end': # do rescale toward end
            p_sym, details = self.do_rescale_toward_end(self.neuron)

        if per == 'sliding general': # do sliding general
            p_sym, details = self.do_sliding_general(self.neuron)
        if per == 'sliding certain in distance': # do sliding in certain distance
            p_sym, details = self.do_sliding_certain_distance(self.neuron)
        if per == 'sliding for branching node': # do sliding only for branch
            p_sym, details = self.do_sliding_branch(self.neuron)
        if per == 'sliding for branching node certain distance': # do sliding only for branch
            p_sym, details = self.do_sliding_branch_certain_distance(self.neuron)

        if per == 'stretching vertical':
            p_sym, details = self.do_vertical_stretching(self.neuron)
        if per == 'stretching horizental':
            p_sym, details = self.do_horizental_stretching(self.neuron)
        if per == 'sinusidal':
            p_sym, details = self.do_sinusidal_wave(self.neuron)
        return p_sym, details

    def do_sinusidal_wave(self, neuron):
        """
        NOT READY
        """
        (branch_index,)  = np.where(neuron.branch_order==2)
        (end_nodes,)  = np.where(neuron.branch_order==0)
        nodes = np.append(branch_index,end_nodes)
        parents = neuron.parent_index_for_node_subset(nodes)
        n = np.floor(nodes.shape[0]*np.random.rand()).astype(int)
        node_index = nodes[n]
        parent_index = parents[n]
        hight = np.exp(np.random.normal() * self.sinusidal_hight)

        neuron.sinudal(node_index, parent_index, hight, n_vertical, n_horizental)
        details = [0,0,0]
        details[0] = node_index
        details[1] = parent_index
        details[2] = hight
        details[3] = n_vertical
        details[4] = n_horizental
        p_sym = 1
        return p_sym, details

    def do_vertical_stretching(self, neuron):
        (branch_index,)  = np.where(neuron.branch_order==2)
        (end_nodes,)  = np.where(neuron.branch_order==0)
        nodes = np.append(branch_index,end_nodes)
        parents = neuron.parent_index_for_node_subset(nodes)
        n = np.floor(nodes.shape[0]*np.random.rand()).astype(int)
        p = np.exp(np.random.normal() * self.horizental_stretch)
        node_index = nodes[n]
        parent_index = parents[n]
        neuron.vertical_stretch(node_index, parent_index, p)
        details = [0,0,0]
        details[0] = node_index
        details[1] = parent_index
        details[2] = p
        p_sym = 1
        return p_sym, details

    def do_horizental_stretching(self, neuron):
        (branch_index,)  = np.where(neuron.branch_order==2)
        (end_nodes,)  = np.where(neuron.branch_order==0)
        nodes = np.append(branch_index,end_nodes)
        parents = neuron.parent_index_for_node_subset(nodes)
        n = np.floor(nodes.shape[0]*np.random.rand()).astype(int)
        p = np.exp(np.random.normal() * self.horizental_stretch)
        node_index = nodes[n]
        parent_index = parents[n]
        neuron.horizental_stretch(node_index, parent_index, p)
        details = [0,0,0]
        details[0] = node_index
        details[1] = parent_index
        details[2] = p
        p_sym = 1
        return p_sym, details

    def select_proposal(self):
        (I,) = np.where(self._consum_prob >= np.random.random_sample(1,))
        return self.p_list[min(I)]

    def do_sliding_certain_distance(self ,neuron):
        branch_node = neuron.get_random_branching_node()
        details = [0,0]
        if branch_node.type is not 'empty':
            if(np.random.rand()<.5):
                child = branch_node.children[0]
            else:
                child = branch_node.children[1]
            child_of_branching_node_index = neuron.get_index_for_no_soma_node(child)
            (I,) = np.where(~ np.isnan(neuron.connection[:, child_of_branching_node_index]))
            I = np.append(I, neuron.get_index_for_no_soma_node(branch_node))
            order_one_node = neuron.get_random_order_one_node_not_in_certain_index(I)
            if order_one_node.type is not 'empty':
                order_one_node_index = neuron.get_index_for_no_soma_node(order_one_node)
                if(LA.norm(neuron.location[:,order_one_node_index] - neuron.location[:,child_of_branching_node_index],2) < self.sliding_limit):
                    neuron.slide(child_of_branching_node_index, order_one_node_index)
                    details[0] = child_of_branching_node_index
                    details[1] = neuron.get_index_for_no_soma_node(branch_node)
        p_sym = 1
        return p_sym, details

    def do_sliding_branch_certain_distance(self ,neuron):
        """
        It selects two positions in the neuron, one branching point and one order-one node, and cut one of the segments of the branching point and translate the whole segment (and all of
        its dependency) to the order-one node.
        """
        branch_node = neuron.get_random_branching_node()
        details = [0,0]
        if branch_node.type is not 'empty':
            if(np.random.rand()<.5):
                child = branch_node.children[0]
            else:
                child = branch_node.children[1]
            child_of_branching_node_index = neuron.get_index_for_no_soma_node(child)
            (I,) = np.where(~ np.isnan(neuron.connection[:, child_of_branching_node_index]))
            I = np.append(I, neuron.get_index_for_no_soma_node(branch_node))
            A = deepcopy(neuron.location)
            A[0,:] = A[0,:] - neuron.location[0, neuron.get_index_for_no_soma_node(branch_node)]
            A[1,:] = A[1,:] - neuron.location[1, neuron.get_index_for_no_soma_node(branch_node)]
            A[2,:] = A[2,:] - neuron.location[2, neuron.get_index_for_no_soma_node(branch_node)]
            (J,) = np.where(LA.norm(A,axis = 0) > self.sliding_limit)
            I = np.append(I,J)
            order_one_node = neuron.get_random_order_one_node_not_in_certain_index(I)
            if order_one_node.type is not 'empty':
                order_one_node_index = neuron.get_index_for_no_soma_node(order_one_node)
                if(LA.norm(neuron.location[:,order_one_node_index] - neuron.location[:,child_of_branching_node_index],2) < self.sliding_limit):
                    neuron.slide(child_of_branching_node_index, order_one_node_index)
                    details[0] = child_of_branching_node_index
                    details[1] = neuron.get_index_for_no_soma_node(branch_node)
        p_sym = 1
        return p_sym, details

    def do_ext_red(self ,neuron):
        L = neuron.ext_red_list
        (op, node_index) = self.select_non_zero_element_with_soma(L[0:2,:])
        p1 = neuron.p_ext_red_whole()
        if(op == 0): # extend the neuron by adding one node to the node
            details, p_sym = self.extend_node(neuron, node_index)
        if(op == 1): # remove the node
            details, p_sym = self.remove_node(neuron, node_index)
        if(op == 2): # extend soma
            details, p_sym = self.extend_soma(neuron)

        p_sym = p_sym*(p1/neuron.p_ext_red_whole())
        return p_sym, details

    def do_ext_red_end_points(self ,neuron):
        a = neuron.ext_red_list[1:3,:]
        p1 = neuron.p_ext_red_end_point()
        (op, node_index) = self.select_non_zero_element_without_soma(a)
        if(op == 0): # remove the end point
            details, p_sym = self.remove_node(neuron, node_index)
        if(op == 1): # extend the end point
            details, p_sym = self.extend_node(neuron, node_index)
        if(op == 2):
            details = []
            p_sym = 1

        p_sym = p_sym*p1/(neuron.p_ext_red_end_point())
        return p_sym, details

    def do_sliding_branch(self ,neuron):
        """
        It selects two positions in the neuron, one branching point and one order-one node, and cut one of the segments of the branching point and translate the whole segment (and all of
        its dependency) to the order-one node.
        """
        branch_node = neuron.get_random_branching_node()
        details = [0,0]
        if branch_node.type is not 'empty':
            if(np.random.rand()<.5):
                child = branch_node.children[0]
            else:
                child = branch_node.children[1]
            child_of_branching_node_index = neuron.get_index_for_no_soma_node(child)
            (I,) = np.where(~ np.isnan(neuron.connection[:, child_of_branching_node_index]))
            I = np.append(I, neuron.get_index_for_no_soma_node(branch_node))
            order_one_node = neuron.get_random_order_one_node_not_in_certain_index(I)
            if order_one_node.type is not 'empty':
                order_one_node_index = neuron.get_index_for_no_soma_node(order_one_node)
                neuron.slide(child_of_branching_node_index, order_one_node_index)
                details[0] = child_of_branching_node_index
                details[1] = neuron.get_index_for_no_soma_node(branch_node)
        p_sym = 1
        return p_sym, details

    def do_add_remove_node(self ,neuron):
        """
        Add or remove a node in the neuron. Adding can be done by selecting one random node in the tree
        and add a node as its parent. Removing can be done by selecting a random order one node.
        """
        random_node, state = self.get_random_element_for_add_remove(neuron)
        if state is 'add':
            neuron.add_extra_node(random_node)
        if state is 'remove':
            neuron.remove_extra_node(random_node)
        return 1, 1

    def do_location(self ,neuron):
        if(neuron.is_soma()): # To make sure that there is at least one node in the no_soma list
            index = 0
            displace  = [0, 0, 0]
        else:
            index = neuron.choose_random_node_index()
            displace = self.location * np.random.normal(size = 3)
            neuron.change_location(index,displace)
        p_sym = 1
        return p_sym, [index, displace[0], displace[1], displace[2]]

    def do_location_toward_end_nodes(self ,neuron):
        details = [0,0,0,0]
        if(~neuron.is_soma()): # To make sure that there is at least one node in the no_soma list
            index = neuron.choose_random_node_index()
            displace = self.location_toward_cte * np.random.normal(size = 3)
            neuron.change_location_toward_end_nodes(index,displace)
            details[0] = index
            details[1] = displace[0]
            details[2] = displace[1]
            details[3] = displace[2]
        p_sym = 1
        return p_sym, details

    def do_location_important(self, neuron):
        index = neuron.get_index_for_no_soma_node(neuron.get_random_branching_or_end_node())
        displace = self.location_important * np.random.normal(size = 3)
        neuron.change_location_important(index,displace)
        p_sym = 1
        return p_sym, [index, displace[0], displace[1], displace[2]]

    def do_diameter(self ,neuron):
        if(neuron.is_soma()): # To make sure that there is at least one node in the no_soma list
            index = 0
            ratio = 1
        else:
            index = neuron.choose_random_node_index()
            ratio = np.power(2,0.1*np.random.normal(0, 1, 1))
            neuron.change_diameter(index,ratio)
        return 1, [index, ratio]

    def do_rotation_general(self, neuron, kappa):
        matrix = self.random_unitary_basis(kappa)
        node = neuron.get_random_no_soma_node()
        neuron.rotate(node, matrix)
        details = [0,0]
        details[0] = node
        details[1] = matrix
        p_sym = 1
        return p_sym, details

    def do_rotation_branching(self, neuron, kappa):
        node = neuron.get_random_branching_node()
        details = [0,0]
        if node.type is 'empty':
            details[0] = neuron.get_random_no_soma_node()
            details[1] = np.eye(3)

        else:
            node = node.children[np.floor(2*np.random.rand()).astype(int)]
            matrix = self.random_unitary_basis(kappa)
            neuron.rotate_from_branch(node, matrix)
            details[0] = node
            details[1] = matrix
        p_sym = 1
        return p_sym, details

    def do_sliding_general(self ,neuron):
        details = [0,0]
        cutting_node = neuron.get_random_no_soma_node()
        if(cutting_node.parent.type is not 'soma'):
            cutting_node_index = neuron.get_index_for_no_soma_node(cutting_node)
            (I,) = np.where(~ np.isnan(neuron.connection[:, cutting_node_index]))
            attaching_node = neuron.get_random_non_branch_node_not_in_certain_index(I)
            if attaching_node.type is not 'empty':
                attaching_node_index = neuron.get_index_for_no_soma_node(attaching_node)
                details[0] = cutting_node_index
                details[1] = neuron.get_index_for_no_soma_node(cutting_node.parent)
                neuron.slide(cutting_node_index, attaching_node_index)
        p_sym = 1
        return p_sym, details

    def do_rescale_toward_end(self ,neuron):
        details = [0,0]
        node = neuron.get_random_no_soma_node()
        details[0] = node
        re = np.exp(np.random.normal(self.rescale_value))
        details[1] = re
        p_sym = 1
        neuron.rescale_toward_end(node, re)
        return p_sym, details

    def undo_MCMC(self, per, details):
        """
        when per == 0, details[0] is 'ext' of 'remove'. If it is 'ext', then details[1] is node_index.
        if it is 'remove', details[1] = parent, details[2] = location, details[3] = ratio
        """
        if per == 'extension/reduction': # undo extension/reduction
            if(len(details) !=0):
                if(details[0] == 'ext'):
                    self.undo_ext(self.neuron, details[1])
                if(details[0] == 'remove'):
                    self.undo_red(self.neuron, details[1], details[2], details[3])
        if per == 'extension/reduction end points':
            if(len(details) !=0):
                if(details[0] == 'ext'):
                    self.undo_ext(self.neuron, details[1])
                if(details[0] == 'remove'):
                    self.undo_red(self.neuron, details[1], details[2], details[3])

        if per == 'location': # undo location
            if( ~ self.neuron.is_soma()):
                self.undo_location(self.neuron, details[0], details[1], details[2], details[3]) # this function makes a location perturbation on the neuron
        if per == 'location for important point': # undo location
            if( ~ self.neuron.is_soma()):
                self.undo_location_important(self.neuron, details[0], details[1], details[2], details[3]) # this function makes a location perturbation on the neuron
        if per == 'location toward end':
            if( ~ self.neuron.is_soma()):
                self.undo_location_toward_end_nodes(self.neuron, details[0], details[1], details[2], details[3])

        if per == 'diameter': # undo diameter
            if( ~ self.neuron.is_soma()): # To make sure that there is at least one node in the no_soma list
                self.undo_diameter(self.neuron, details[0], details[1])

        if per == 'rotation for any node':
            self.undo_rotation(self.neuron, details[0], details[1] )
        if per == 'rotation for branching':
            self.undo_rotation_from_branch(self.neuron, details[0], details[1] )

        if per == 'sliding certain in distance': # undo sliding in certain distance
            if(details[0] != 0):
                self.undo_sliding(self.neuron, details[0], details[1])
        if per == 'sliding for branching node': # undo sliding for branch
            if(details[0] != 0):
                self.undo_sliding(self.neuron, details[0], details[1])
        if per == 'sliding general': # undo sliding general
            if(details[0] != 0):
                self.undo_sliding_general(self.neuron, details[0], details[1])
        if per == 'sliding for branching node certain distance': # do sliding only for branch
            if(details[0] != 0):
                self.undo_sliding(self.neuron, details[0], details[1])

        if per == 'rescale toward end':
            self.undo_rescale_toward_end(self.neuron, details[0],details[1])

        if per == 'stretching vertical':
            self.undo_vertical_stretching(self.neuron, details[0], details[1], details[2])
        if per == 'stretching horizental':
            self.undo_horizental_stretching(self.neuron, details[0], details[1], details[2])

        if per == 'sinusidal':
            self.undo_sinusidal_wave(self.neuron, details[0], details[1], details[2], details[3], details[4])

    def undo_sinusidal_wave(self, neuron, node, parent, hight, n_vertical, n_horizental):
        neuron.sinudal(node_index, parent_index, -hight, n_vertical, n_horizental)

    def undo_location(self, neuron, index, x, y, z):
        neuron.change_location(index, - np.array([x,y,z]))

    def undo_location_toward_end_nodes(self, neuron, index, x, y, z):
        neuron.change_location_toward_end_nodes(index, - np.array([x,y,z]))

    def undo_location_important(self, neuron, index, x, y, z):
        neuron.change_location_important(index, - np.array([x,y,z]))

    def undo_diameter(self, neuron, index, ratio):
        neuron.change_diameter(index, 1.0 / ratio)

    def undo_ext(self, neuron, index_node):
        neuron.remove_node(index_node)

    def undo_red(self, neuron, parent, location, ratio):
        neuron.extend_node(parent, location, ratio)

    def undo_rotation(self, neuron, node, matrix):
        neuron.rotate(node, inv(matrix))

    def undo_rotation_from_branch(self, neuron, node, matrix):
        neuron.rotate_from_branch(node, inv(matrix))

    def undo_sliding(self, neuron, child_of_branching_node_index, order_one_node_index):
        neuron.slide(child_of_branching_node_index, order_one_node_index)

    def undo_sliding_general(self, neuron,child_of_branching_node_index, order_one_node_index):
        neuron.slide(child_of_branching_node_index, order_one_node_index)

    def undo_rescale_toward_end(self, neuron, node, rescale):
        neuron.rescale_toward_end(node, 1./rescale)

    def undo_vertical_stretching(self, neuron, node, parent, scale):
        neuron.vertical_stretch(node, parent, 1./scale)

    def undo_horizental_stretching(self, neuron, node, parent, scale):
        neuron.horizental_stretch(node, parent, 1./scale)

    def set_measure(self, m):
        self.measure = m
        self.list_features = m['mean'].keys()
        self.mean_measure = np.array([])
        self.variance_measure = np.array([])
        self.sd_measure = np.array([])
        for ind in self.list_features:
            self.mean_measure = np.append(self.mean_measure, float(m['mean'][ind]))
            self.variance_measure = np.append(self.variance_measure, float(m['variance'][ind]) ** 2)
            self.sd_measure = np.append(self.sd_measure, float(m['variance'][ind]))
        self.trend = np.zeros([len(m['mean']),self.ite])
        self.trend_normal = np.zeros([len(m['mean']),self.ite])

    def set_probability(self, list_values):
        """
        set the probability for perturbation
        list_values : dict

        """
        l = sum(list_values.values())
        for i in list_values.keys():
            list_values[i] = list_values[i]/l
        self.p_prob = np.array(list_values.values())
        self.p_list = list_values.keys()
        self._consum_prob = np.zeros(len(list_values.keys()))
        for i in range(self.p_prob.shape[0]):
            self._consum_prob[i] = sum(self.p_prob[:i+1])

    def set_real_neuron(self, neuron, m):
        """
        Set the desire features by the features of given neuron. No dependency.

        """
        self.list_features = m['hist_range'].keys()
        self.n_features = len(self.list_features)
        self.mean_hist = {}
        self.hist_range = {}
        self.variance_measure = {}
        self.std_measure = {}
        for k in range(len(self.list_features)):
            name = self.list_features[k]
            a = neuron.features[name]
            a = a[~np.isnan(a)]
            self.hist_range[name] = m['hist_range'][name]
            a = np.histogram(a,bins = self.hist_range[name])[0].astype(float)
            #a = a/a.sum()
            if(self.hist_range[name].shape[0] == 1):
                a = neuron.features[name]
            self.mean_hist[name] = a
            self.variance_measure[name] = m['hist_variance'][name]**2
            self.std_measure[name] = m['hist_variance'][name]
        self.trend = np.zeros([len(m['hist_range']),self.ite])
        self.trend_normal = np.zeros([len(m['hist_range']),self.ite])

    def pdf_normal(self , x, dim):
        """
        Return the probability density at the point x of a normal distribution with mean = 0
        and variance = s
        """
        rv = multivariate_normal(np.zeros(dim), self.var*np.eye(dim))
        return rv.pdf(x)
        # should notice to the dimentionality of the constant return (self.cte_gauss/s)*np.power(np.e,-(x*x).sum()/(s*s))

    def normal(self, dim):
        random_point = np.random.normal(0, self.var, dim)
        rv = multivariate_normal(np.zeros(dim), self.var*np.eye(dim))
        pdf = rv.pdf(random_point)
        return random_point, pdf

    def random_vector(self, mean, var):
        vec = np.random.normal(size = 3)
        vec = vec/LA.norm(vec,2)
        l = -1
        while(l<0):
            l = mean + var * np.random.normal()
        vec = vec*l
        return vec

    def get_random_element_for_add_remove(self, neuron):
        (ind1,) = np.where(neuron.branch_order[neuron.n_soma:] == 1)
        whole = len(neuron.nodes_list) - neuron.n_soma
        total_number = len(ind1) + whole
        a = np.floor(total_number * np.random.rand())
        if(a < whole):
            random_node = neuron.nodes_list[neuron.n_soma + a]
            state = 'add'
        else:
            random_node = neuron.nodes_list[ind1[a - whole]]
            state = 'remove'
        return total_number ,random_node, state

    def random_rotation(self, vector, mu, kappa, n):
        """
        input: mu, kappa, n `float64`

        Return three vectors: the first one is close to the given vector; these three vectors make a complete
        set of orthogonal space for 3D space.
        The first vector is choosen accroding to a distribution for the
        phi (the angle between the given vector and choosen one)
        and unifor distribution for the theta (the angle of projection of the choosen vector over the orthogonal plane)
        the phi angle comes from von Mises distribution.
        """
        vector = vector/LA.norm(vector,2)
        a = np.random.normal(0, 1, 3)
        a = a - sum(a*vector)*vector
        a = a/LA.norm(a,2)
        phi = np.random.vonmises(mu, kappa, 1)
        normal_vec = np.sin(phi)*a + np.cos(phi)*vector
        length = np.random.chisquare(n,1)/n
        random_point = length*normal_vec
        pdf = (.5/np.pi)*(chi2.pdf(n*length,n)*n)*(vonmises.pdf(np.cos(phi), kappa))
        return random_point, pdf

    def pdf_random_rotation(self, x, v, mu, kappa, n):
        """
        Gives back the probability of observing the vector x, such that its angle with v is coming from a Von Mises
        distribution with k = self.kappa and its length coming form chi squared distribution with the parameter n.
        """
        v = v/LA.norm(v,2)
        x = x/LA.norm(x,2)
        ang = sum(v*x)
        return (.5/np.pi)*(chi2.pdf(n*LA.norm(x,2),n)*n)*(vonmises.pdf(ang, kappa))

    def unifrom(self,size):
        return size*(2*np.random.rand(1,3)-1)

    def random_unitary_basis(self, kappa):
        Ax1 = self.random_2d_rotation_in_3d('x', kappa)
        Ay1 = self.random_2d_rotation_in_3d('y', kappa)
        Az1 = self.random_2d_rotation_in_3d('z', kappa)
        Ax2 = self.random_2d_rotation_in_3d('x', kappa)
        Ay1 = self.random_2d_rotation_in_3d('y', kappa)
        Az1 = self.random_2d_rotation_in_3d('z', kappa)
        A = np.dot(np.dot(Ax1,Ay1),Az1)
        B = np.dot(np.dot(Az1,Ay1),Ax1)
        return np.dot(A,B)

    def random_2d_rotation_in_3d(self, axis, kappa):
        theta = np.random.vonmises(0, kappa, 1)
        A = np.eye(3)
        if axis is 'z':
            A[0,0] = np.cos(theta)
            A[1,0] = np.sin(theta)
            A[0,1] = - np.sin(theta)
            A[1,1] = np.cos(theta)
            return A
        if axis is 'y':
            A[0,0] = np.cos(theta)
            A[2,0] = np.sin(theta)
            A[0,2] = - np.sin(theta)
            A[2,2] = np.cos(theta)
            return A
        if axis is 'x':
            A[1,1] = np.cos(theta)
            A[2,1] = np.sin(theta)
            A[1,2] = - np.sin(theta)
            A[2,2] = np.cos(theta)
            return A

    def vector_par(self, node):
        if node.parent.type is 'soma':
            return np.array([1.,.0,.0])
        else:
            return self.neuron.xyz(node)

    def extend_soma(self, neuron):
        #r, pdf_r = self.normal(self.mean_ratio_diameter,1)
        #l, pdf_l = self.random_rotation(np.array([1.,.0,.0]), self.mu, self.kappa, self.n_chi)

        #r, pdf_r = self.normal(self.mean_ratio_diameter,1)
        r = 0
        l = self.random_vector(self.mean_len_to_parent, self.var_len_to_parent)
        n = neuron.extend_node('soma', l , np.exp(r))
        details = [0,0]
        details[0] = 'ext'
        details[1] = n
        p_sym = self.ratio_red_to_ext
        return details, p_sym

    def extend_node(self, neuron, node_index):
        node = neuron.nodes_list[node_index]
        #r, pdf_r = self.normal(self.mean_ratio_diameter,1)
        #par_loc = self.vector_par(self.neuron.nodes_list[node_index])
        #l, pdf_l = self.random_rotation(par_loc, self.mu, self.kappa, self.n_chi)
        #r, pdf_r = self.normal(self.mean_ratio_diameter,1)
        r = 0
        l = self.random_vector(self.mean_len_to_parent, self.var_len_to_parent)
        n = neuron.extend_node(node,l,np.exp(r))
        details = [0,0]
        details[0] = 'ext'
        details[1] = n
        p_sym = self.ratio_red_to_ext
        return details, p_sym

    def remove_node(self, neuron, node_index):
        par_loc = self.vector_par(neuron.nodes_list[node_index])
        p, l, r = neuron.remove_node(node_index)
        details = [0,0,0,0]
        details[0] = 'remove'
        details[1] = p
        details[2] = l
        details[3] = r
        #p_sym = norm.pdf(np.log(r),0,self.mean_ratio_diameter)*self.pdf_random_rotation(l, par_loc, self.mu, self.kappa, self.n_chi)
        #p_sym = self.pdf_normal(np.log(r), self.mean_ratio_diameter, 1)*self.pdf_normal(l,self.mean_loc, 3)
        p_sym = 1./(self.ratio_red_to_ext)
        return details, p_sym

    def minus_log_prob_neuron(self, neuron):
        er = 0.0
        error_vec = np.array([])
        error_vec_normal = np.array([])
        for k in range(len(self.list_features)):
            raw = self.neuron.features[self.list_features[k]] - self.mean_measure[k]
            a = (raw) ** 2
            E = (np.array(a,ndmin = 1)/self.variance_measure[k]).mean()
            error_vec_normal = np.append(error_vec_normal, (np.array(raw,ndmin = 1)/self.sd_measure[k]).mean())
            error_vec = np.append(error_vec, E)
            if(self.verbose == 1):
                print self.list_features[k]
                print E
            er += E
        return er, error_vec, error_vec_normal

    def kl_distance(self, neuron):
        er = 0.0
        error_vec = np.zeros(self.n_features)
        error_vec_normal = np.zeros(self.n_features)
        for k in range(self.n_features):
            if(self.verbose == 1):
                print self.list_features[k]
            name = self.list_features[k]
            feature = neuron.features[name]
            feature = feature[~np.isnan(feature)]
            if(self.hist_range[name].shape[0]>1):
                hist_fea = np.histogram(feature,bins = self.hist_range[name])[0].astype(float)
                diff_fea = hist_fea - self.mean_hist[name]
                E = ((diff_fea ** 2)/self.variance_measure[name]).mean()
                error_vec_normal[k] = (diff_fea/self.std_measure[name]).mean()
            else:
                diff_fea = feature - self.mean_hist[name]
                E = (diff_fea ** 2)/self.variance_measure[name]
                E = E[0]
                error_vec_normal[k] = (diff_fea/self.std_measure[name]).mean()
            error_vec[k] = E
            if(self.verbose == 1):
                print self.list_features[k]
                print E
            er += E
        return er, error_vec, error_vec_normal

    def accept_proposal(self, a):
        return (a > np.random.random_sample(1,))[0]

    def select_non_zero_element_with_soma(self, matrix):
        soma = np.zeros([1,matrix.shape[1]])
        soma[0,0] = 1
        m = np.append(matrix,soma,axis = 0)
        a, b = self.select_element_prob_matrix(m)
        return a , b

    def select_non_zero_element_without_soma(self, matrix):
        if(matrix.sum()!=0):
            a, b = self.select_element_prob_matrix(matrix)
        else:
            a = 2
            b = 0
        return a , b

    def select_element_prob_matrix(self, matrix):
        """
        input
        -----
            matrix : 2d array. It should be a probability matrix; i.e non-negative and

        output
        ------
            the index one element
        """
        [a,b] = matrix.shape
        rematrix = matrix/matrix.sum()
        m = rematrix.reshape([a*b])
        r = np.random.choice(a*b, 1, p =m)[0]
        y = np.remainder(r, b)
        x = (r-y)/b
        return x,y

    def show_MCMC(self,start,size_x,size_y):
        plt.figure(figsize=(size_x,size_y))
        plt.subplot(1,3,1)
        plt.plot(sum(self.trend[:,start:],0));
        plt.subplot(1,3,2)
        plt.plot(np.transpose(self.trend_normal[:,start:]));
        #plt.legend(self.list_features,bbox_to_anchor=(2.1,1.1))
        plt.subplot(1,3,3)
        plt.plot(np.transpose(self.trend[:,start:]));
        plt.legend(self.list_features,bbox_to_anchor=(2.1,1.1))
