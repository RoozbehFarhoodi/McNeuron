"""Collection of useful data transforms."""

# Imports
import numpy as np


def get_leaves(nodes, parents):
    """
    Compute the list of leaf nodes.

    Parameters
    ----------
    nodes: list
        list of all nodes in the tree
    parents: list
        list of parents for each node

    Returns
    -------
    leaves: list
        sorted list of leaf nodes
    """
    leaves = np.sort(list(set(nodes) - set(parents)))
    return leaves


def encode_prufer(parents, verbose=0):
    """
    Convert the parents sequence to a prufer sequence.

    Parameters
    ----------
    parents: list
        list of parents for each node
    verbose: bool
        default is False

    Returns
    -------
    prufer: list
        corresponding prufer sequence
    """
    n_nodes = len(parents)
    nodes = range(n_nodes)

    prufer = list()
    for n in range(n_nodes - 2):

        # Recalculate all the leaves
        leaves = get_leaves(nodes, parents)
        if verbose:
            print 'leaves', leaves

        # Add the parent of the lowest numbered leaf to the sequence
        leaf_idx = np.where(nodes == leaves[0])[0][0]
        prufer.append(parents[leaf_idx])
        if verbose:
            print 'prufer', prufer

        # Remove the lowest numbered leaf and its corresponding parent
        del nodes[leaf_idx]
        del parents[leaf_idx]

        if verbose:
            print 'nodes', nodes
            print 'parents', parents
            print 60*'-'

    return prufer


def decode_prufer(prufer, verbose=0):
    """
    Convert the prufer sequence to a parents sequence.

    Parameters
    ----------
    prufer: list
        prufer sequence
    verbose: bool
        default is False

    Returns
    -------
    parents: list
        corresponding list of parents for each node
    """
    n_nodes = len(prufer) + 2
    n_prufer = len(prufer)
    nodes = range(n_nodes)
    parents = -1 * np.ones(n_nodes)

    for n in range(n_prufer):
        if verbose:
            print nodes
            print prufer
        leaves = list(get_leaves(nodes, prufer))
        k = leaves[0]
        j = prufer[0]

        if k == 0:
            k = leaves[1]

        if verbose:
            print k, j
        parents[k] = j

        leaf_idx = np.where(nodes == k)[0][0]
        del nodes[leaf_idx]
        del prufer[0]

        if verbose:
            print 60*'-'

    parents[nodes[1]] = nodes[0]
    return list(parents.astype(int))
