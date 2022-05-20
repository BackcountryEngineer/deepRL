import imp
from sum_tree import SumTree
import numpy as np

class PrioritizedMemory:
    def __init__(self, capacity):
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        
        self.PER_b_increment_per_sampling = 0.001
    
        self.absolute_error_upper = 1.0  # clipped abs error
        
        self.tree = SumTree(capacity)

    def add(self, experience):
        max_priority = self.tree.max_priority

        if max_priority < 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = self.tree.min_priority / self.tree.total_priority
        max_weight = np.power(p_min * n, -self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a,b)
            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority

            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b / max_weight)
            b_idx[i] = index
            
            memory_b.append(data)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

        
