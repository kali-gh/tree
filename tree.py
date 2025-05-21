from binary_tree import BinaryTree

from libs import Node
from typing import List, Dict
from copy import deepcopy

class Tree:

    def __init__(self, nodes):
        self._nodes : Dict[int : Node] = {}
        self._bt : BinaryTree = BinaryTree()

        for n in nodes:
            self._bt.add_edge(n.parent_id, n.id)
            self._nodes[n.id] = deepcopy(n)

