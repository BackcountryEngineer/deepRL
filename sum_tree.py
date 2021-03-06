import numpy as np

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.curr_max_priority = np.NINF
        self.curr_min_priority = np.inf

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data

        self.update(tree_index, priority)

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self.curr_max_priority = np.max([self.curr_max_priority, priority])
        self.curr_min_priority = np.min([self.curr_min_priority, priority])

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def max_priority(self):
        return self.curr_max_priority

    @property
    def min_priority(self):
        return self.curr_min_priority

    @property
    def total_priority(self):
        return self.tree[0]
