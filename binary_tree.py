import numpy as np
import copy

from dataclasses import dataclass

@dataclass
class Node:
    id : int
    data : np.array

class BinaryTree:

    _adjs = {}

    @property
    def nodes(self):
        return copy.deepcopy(list(self._adjs.keys()))

    def add_edge(self, parent, child):
        if self._adjs.get(parent) is None:
            self._adjs[parent] = list()
        else:
            pass

        self._adjs[parent].append(child)

        if self._adjs.get(child) is None: # make sure the child is registered
            self._adjs[child] = list()
        else:
            pass

    def is_leaf(self, node_id):
        children = self._adjs.get(node_id)
        assert children is not None, "node children cannot be none when it exists, check node is in the tree"
        return len(children) == 0

    def is_internal(self, node_id):
        return not self.is_leaf(node_id) and not node_id == 0 and not node_id is None

    def get_all_children(self, node_id):

        all_children = []

        node_ids_to_process = [node_id]

        while len(node_ids_to_process) != 0:
            cur_node_id = node_ids_to_process.pop(0)
            children = self._adjs[cur_node_id]
            all_children.extend(children)
            node_ids_to_process.extend(children)

        return all_children

    def prune(self, node_id):
        assert self.is_internal(node_id), "must be an internal node to prune"

        import copy

        new_adjs = copy.deepcopy(self._adjs)

        node_ids_to_pop = [node_id]
        node_ids_to_pop.extend(self.get_all_children(node_id))

        for node_id in node_ids_to_pop:
            new_adjs.pop(node_id)

        self._adjs = new_adjs.copy()
        print(f"pruned {len(node_ids_to_pop)} nodes")

    def get_leaves(self):
        all_nodes = self.nodes

        leaves = []

        for n in all_nodes:
            if self.is_leaf(n):
                leaves.append(n)
            else:
                pass

        return leaves

    def get_parent(self, node_id):
        parent = None
        for k, v in self._adjs.items():
            if node_id in v:
                return k
            else:
                pass
        return parent

    def get_arms(self):

        leaves = self.get_leaves()
        arms = []

        for leaf in leaves:
            arm = []
            cur_node = leaf
            while cur_node is not None:
                arm.append(cur_node)
                cur_node = self.get_parent(cur_node)
            arm.reverse()
            arms.append(arm)

        return arms

