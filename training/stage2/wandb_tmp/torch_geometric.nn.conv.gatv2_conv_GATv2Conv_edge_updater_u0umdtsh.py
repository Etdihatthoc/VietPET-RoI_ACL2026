import typing
from typing import Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils import is_sparse
from torch_geometric.typing import Size, SparseTensor

from torch_geometric.nn.conv.gatv2_conv import *


from typing import List, NamedTuple, Optional, Union

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.index import ptr2index
from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.typing import SparseTensor


class CollectArgs(NamedTuple):
    x_j: Tensor
    x_i: Tensor
    edge_attr: Optional[Tensor]
    index: Tensor
    ptr: Optional[Tensor]
    dim_size: Optional[int]


def edge_collect(
    self,
    edge_index: Union[Tensor, SparseTensor],
    x: PairTensor,
    edge_attr: OptTensor,
    size: List[Optional[int]],
) -> CollectArgs:

    i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

    # Collect special arguments:
    if isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            adj_t = edge_index
            if adj_t.layout == torch.sparse_coo:
                edge_index_i = adj_t.indices()[0]
                edge_index_j = adj_t.indices()[1]
                ptr = None
            elif adj_t.layout == torch.sparse_csr:
                ptr = adj_t.crow_indices()
                edge_index_j = adj_t.col_indices()
                edge_index_i = ptr2index(ptr, output_size=edge_index_j.numel())
            else:
                raise ValueError(f"Received invalid layout '{adj_t.layout}'")
            if edge_attr is None:
                _value = adj_t.values()
                edge_attr = None if _value.dim() == 1 else _value

        else:
            edge_index_i = edge_index[i]
            edge_index_j = edge_index[j]

            ptr = None
            if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
                if i == 0 and edge_index.is_sorted_by_row:
                  (ptr, _), _ = edge_index.get_csr()
                elif i == 1 and edge_index.is_sorted_by_col:
                  (ptr, _), _ = edge_index.get_csc()

    elif isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        edge_index_i, edge_index_j, _value = adj_t.coo()
        ptr, _, _ = adj_t.csr()
        if edge_attr is None:
            edge_attr = None if _value is None or _value.dim() == 1 else _value

    else:
        raise NotImplementedError

    # Collect user-defined arguments:
    # (1) - Collect `x_j`:
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        _x_0, _x_1 = x[0], x[1]
        if isinstance(_x_0, Tensor):
            self._set_size(size, 0, _x_0)
            x_j = self._index_select(_x_0, edge_index_j)
        else:
            x_j = None
        if isinstance(_x_1, Tensor):
            self._set_size(size, 1, _x_1)
    elif isinstance(x, Tensor):
        self._set_size(size, j, x)
        x_j = self._index_select(x, edge_index_j)
    else:
        x_j = None
    # (2) - Collect `x_i`:
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        _x_0, _x_1 = x[0], x[1]
        if isinstance(_x_0, Tensor):
            self._set_size(size, 0, _x_0)
        if isinstance(_x_1, Tensor):
            self._set_size(size, 1, _x_1)
            x_i = self._index_select(_x_1, edge_index_i)
        else:
            x_i = None
    elif isinstance(x, Tensor):
        self._set_size(size, i, x)
        x_i = self._index_select(x, edge_index_i)
    else:
        x_i = None

    # Collect default arguments:

    index = edge_index_i
    size_i = size[i] if size[i] is not None else size[j]
    size_j = size[j] if size[j] is not None else size[i]
    dim_size = size_i

    return CollectArgs(
        x_j,
        x_i,
        edge_attr,
        index,
        ptr,
        dim_size,
    )


def edge_updater(
    self,
    edge_index: Union[Tensor, SparseTensor],
    x: PairTensor,
    edge_attr: OptTensor,
    size: Size = None,
) -> Tensor:

    mutable_size = self._check_input(edge_index, size)

    kwargs = self.edge_collect(
        edge_index,
        x,
        edge_attr,
        mutable_size,
    )

    # Begin Edge Update Forward Pre Hook #######################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._edge_update_forward_pre_hooks.values():
            hook_kwargs = dict(
                x_j=kwargs.x_j,
                x_i=kwargs.x_i,
                edge_attr=kwargs.edge_attr,
                index=kwargs.index,
                ptr=kwargs.ptr,
                dim_size=kwargs.dim_size,
            )
            res = hook(self, (edge_index, size, hook_kwargs))
            if res is not None:
                edge_index, size, hook_kwargs = res
                kwargs = CollectArgs(
                    x_j=hook_kwargs['x_j'],
                    x_i=hook_kwargs['x_i'],
                    edge_attr=hook_kwargs['edge_attr'],
                    index=hook_kwargs['index'],
                    ptr=hook_kwargs['ptr'],
                    dim_size=hook_kwargs['dim_size'],
                )
    # End Edge Update Forward Pre Hook #########################################

    out = self.edge_update(
        x_j=kwargs.x_j,
        x_i=kwargs.x_i,
        edge_attr=kwargs.edge_attr,
        index=kwargs.index,
        ptr=kwargs.ptr,
        dim_size=kwargs.dim_size,
    )

    # Begin Edge Update Forward Hook ###########################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._edge_update_forward_hooks.values():
            hook_kwargs = dict(
                x_j=kwargs.x_j,
                x_i=kwargs.x_i,
                edge_attr=kwargs.edge_attr,
                index=kwargs.index,
                ptr=kwargs.ptr,
                dim_size=kwargs.dim_size,
            )
            res = hook(self, (edge_index, size, hook_kwargs), out)
            out = res if res is not None else out
    # End Edge Update Forward Hook #############################################

    return out