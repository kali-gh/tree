import numpy as np
import math
import copy

from enum import Enum
from dataclasses import dataclass
from typing import List


class Direction(Enum):
    NO_DIRECTION = 0
    LEFT = -1
    RIGHT = 1

@dataclass
class Split:
    splitting_variable : int
    splitting_point : float
    splitting_direction : Direction.NO_DIRECTION


@dataclass
class Node:
    split : Split | None # root node can be none
    data : np.array
    parent_id : int | None # root node can be none
    id : int
    depth : int

    def __str__(self):
        return f"({self.id=},{self.parent_id=})"

    def __repr__(self):
        return self.__str__()

def build_nodes(
        parent_node,
        max_depth,
        cur_id):

    if parent_node.depth >= max_depth:
        return [], cur_id
    else:
        data = parent_node.data
        parent_id = parent_node.id

        splitting_variable, splitting_point, loss = get_optimal_splitting_variable_and_point(data)

        r1 = data[data[:, splitting_variable] <= splitting_point, :]
        r2 = data[data[:, splitting_variable] > splitting_point, :]

        nodes = list()

        split_1 = Split(splitting_variable, splitting_point, Direction.LEFT)
        node_1 = Node(split=split_1, data=r1, parent_id=parent_id, id=cur_id+1, depth=parent_node.depth+1)

        split_2 = Split(splitting_variable, splitting_point, Direction.RIGHT)
        node_2 = Node(split=split_2, data=r2, parent_id=parent_id, id=cur_id+2, depth=parent_node.depth+1)

        nodes.append(node_1)
        nodes.append(node_2)

        return nodes, cur_id+2

def build_tree_nodes(Xy, max_depth):
    root_node = Node(
        split=None,
        data=Xy,
        parent_id=None,
        id=0,
        depth=0
    )

    nodes_to_process = [root_node]
    processed_nodes = []

    cur_id = 0
    while len(nodes_to_process) != 0:
        node_to_process = nodes_to_process.pop(0)

        children, cur_id = build_nodes(parent_node=node_to_process, max_depth=max_depth, cur_id = cur_id)
        processed_nodes.append(node_to_process)

        nodes_to_process.extend(children)

    return processed_nodes

def get_loss_at_split(
        X : np.array,
        y : np.array,
        splitting_variable : int,
        splitting_point : float):
    """
    Gets the l2 loss for the splitting variable at the splitting point given the array X and data y

    :param X: data (MxN)
    :param y: outcome (M,)
    :param splitting_variable: integer index of the variable to split on
    :param splitting_point: floating point to split at
    :return: estimated loss
    """

    Xy = np.hstack([X, y.reshape(-1,1)])

    r1 = Xy[Xy[:, splitting_variable] <= splitting_point, :]
    r2 = Xy[Xy[:, splitting_variable] > splitting_point, :]

    if r1.shape[0] ==0 or r2.shape[0] == 0:
        return math.inf
    else:
        pass

    c1_hat = r1[:, -1].mean()
    c2_hat = r2[:, -1].mean()

    r1_delta = r1[:, -1] - c1_hat
    r1_delta_squared_sum = np.sum(r1_delta**2)

    r2_delta = r2[:, -1] - c2_hat
    r2_delta_squared_sum = np.sum(r2_delta**2)

    loss = r1_delta_squared_sum + r2_delta_squared_sum

    return loss

def get_optimal_splitting_variable_and_point(Xy):
    min_loss = math.inf
    min_loss_splitting_variable = -1
    #min_loss_splitting_point = None

    for splitting_variable in range(Xy.shape[1] - 1):

        losses, xi = get_losses_for_variable(Xy, splitting_variable=splitting_variable)

        variable_min_loss = min(losses)
        if variable_min_loss < min_loss:

            min_loss_idx = np.argmin(losses)
            min_loss_splitting_variable = splitting_variable
            min_loss_splitting_point = float(xi[min_loss_idx])

            min_loss = variable_min_loss
        else:
            pass

    if min_loss_splitting_variable == -1:
        # this can only happen if there is <=1 data point in the leaf, which should never be allowed.
        raise ValueError("No splitting variable found")

    return min_loss_splitting_variable, min_loss_splitting_point, min_loss


def get_losses_for_variable(
        Xy,
        splitting_variable,
        n_granularity=100):
    """
    Gets the losses for a given variable and granularity
    :param Xy: X hstacked with y
    :param splitting_variable: index of variable we are splitting on
    :param n_granularity: granularity in terms of number of datapoints to compute the loss for along splitting_variable axis
    :return: list of losses for the splitting variable (length = n_granularity)
    """

    X = Xy[:, :-1]
    y = Xy[:, -1]

    splitting_point_candidates = np.linspace(min(X[:, splitting_variable]), max(X[:, splitting_variable]), n_granularity)
    losses = []

    for splitting_point in splitting_point_candidates:
        loss = get_loss_at_split(X, y, splitting_variable=splitting_variable, splitting_point=splitting_point)
        losses.append(float(loss))

    return losses, splitting_point_candidates


from libs import Direction

def get_arm_idx_given_candidate_point(cx, tree):
    arms = tree._bt.get_arms()
    found_arm_idx = -1

    for arm_idx, arm in enumerate(arms):

        is_in_arm = True

        for node_id in arm:

            split = tree._nodes[node_id].split

            if split is None:
                continue    # root node
            else:
                pass

            sv_at_li = split.splitting_variable
            sp_at_li = split.splitting_point
            sd_at_li = split.splitting_direction

            if sd_at_li == Direction.LEFT:
                is_in_arm = is_in_arm and (cx[sv_at_li] <= sp_at_li)
            elif sd_at_li == Direction.RIGHT:
                is_in_arm = is_in_arm and (cx[sv_at_li] > sp_at_li)
            else:
                raise ValueError("bad")

            if not is_in_arm:
                break
            else:
                pass

        if is_in_arm:
            found_arm_idx = arm_idx
        else:
            pass

    assert(found_arm_idx != -1)

    return found_arm_idx


def add_arm_indices_to_Xy(Xy, tree):
    a = np.zeros(shape=(Xy.shape[0], 1))
    Xya = np.hstack([Xy, a])

    for m in range(Xy.shape[0]):
        arm_idx = get_arm_idx_given_candidate_point(Xy[m, :-1], tree)
        Xya[m, -1] = arm_idx

    return Xya

def predict_at_cx(Xya, cx, tree):
    arm_idx = get_arm_idx_given_candidate_point(cx, tree)
    arm_stats = get_arm_stats(Xya, tree)

    return arm_stats['y_pred'][arm_idx]

def predict(Xya, tree):
    arm_stats = get_arm_stats(Xya, tree)

    y_pred = []

    for m in range(Xya.shape[0]):
        arm_idx = Xya[m, -1]
        y_pred.append(arm_stats['y_pred'][arm_idx])

    return np.array(y_pred)

def get_arm_stats(Xya, tree):
    arm_stats = {}
    arm_stats['y_pred'] = {}
    arm_stats['count'] = {}
    arm_stats['Q_mT'] = {}

    arms = tree._bt.get_arms()

    for arm_idx, arm in enumerate(arms):

        arm_data = Xya[Xya[:, -1] == arm_idx, :]
        arm_mean = float(arm_data[:, -2].mean())

        arm_stats['y_pred'][arm_idx] = arm_mean

        arm_stats['count'][arm_idx] = arm_data.shape[0]

    for arm_idx, arm in enumerate(arms):
        arm_stats['Q_mT'][arm_idx] = np.sum((arm_data[:, -2] - arm_stats['y_pred'][arm_idx])**2) / arm_stats['count'][arm_idx]
        arm_stats['Q_mT'][arm_idx] = float(arm_stats['Q_mT'][arm_idx])

    return arm_stats

def calculate_cost_complexity(arm_stats, alpha):

    cost_complexity = 0

    abs_T = len(arm_stats['count'].keys())
    print(f"number of arms : {abs_T}")

    for arm_idx in arm_stats['count'].keys():
        cost_complexity +=  arm_stats['count'][arm_idx] * arm_stats['Q_mT'][arm_idx]

    cost_complexity += alpha * abs_T

    return cost_complexity
