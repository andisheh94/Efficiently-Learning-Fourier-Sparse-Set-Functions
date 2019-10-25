import numpy as np
import random


class DecisionTree(object):
    def __init__(self, n_var, n_nodes, max_depth):
        self.n_var = n_var
        self.n_nodes = n_nodes
        self.max_depth = max_depth

        # 0 -> not available
        # 1 -> available
        self.available_nodes = set()
        self.graph = {0: [-1, -1]}
        self.available_nodes.add(0)
        self.labels = {}
        self.leafs = {}

        assert self.n_nodes % 2 == 1, "n_nodes needs to be odd"
        for i in range(0, int((self.n_nodes - 1) / 2)):
            self.add_random_node()

        self.label_graph()
        self.add_reals()

    def label_graph(self):
        available_labels = np.ones(self.n_var)
        node_stack = [0]
        variable_stack = []
        while node_stack:
            node = node_stack[-1]
            if node in self.labels:
                node_stack.pop()
                v = variable_stack.pop()
                available_labels[v] = 1
                continue
            label = np.random.choice(available_labels.nonzero()[0])
            self.labels[node] = label
            variable_stack.append(label)
            available_labels[label] = 0
            print(node, label)
            if node in self.graph and not self.isleaf(node):
                right = self.right(node)
                if right in self.graph:
                    node_stack.append(right)
                left = self.left(node)
                if left in self.graph:
                    node_stack.append(left)

    def add_reals(self):
        for k, v in self.graph.items():
            if self.isleaf(k):
                self.leafs[k] = np.random.uniform(0, 100.0)

    def add_random_node(self):
        node = self.pick_free_node()
        self.available_nodes.remove(node)

        new_node_left = self.left(node)
        new_node_right = self.right(node)
        self.graph[node] = [new_node_left, new_node_right]

        self.graph[new_node_left] = [-1, -1]
        self.graph[new_node_right] = [-1, -1]
        if self.depth(new_node_left) < self.max_depth - 1:
            self.available_nodes.add(new_node_left)
            self.available_nodes.add(new_node_right)

    def pick_free_node(self):
        return random.choice(tuple(self.available_nodes))

    def left(self, node_num):
        return 2 * node_num + 1

    def right(self, node_num):
        return 2 * node_num + 2

    def depth(self, node_num):
        return int(np.log2(node_num + 1))

    def isleaf(self, node_num):
        return sum(self.graph[node_num]) == -2

    def __getitem__(self, key):
        curr = 0
        while not self.isleaf(curr):
            curr_var = self.labels[curr]
            curr = self.graph[curr][key[curr_var]]
        return self.leafs[curr]


if __name__ == '__main__':
    tree = DecisionTree(10, 5, 20)
    print(tree.graph)
    print(tree.labels)
    print(tree.leafs)
