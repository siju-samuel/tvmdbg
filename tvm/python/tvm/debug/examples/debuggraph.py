from __future__ import print_function
#import os
#import inspect
import json

class DebugGraph(object):
  def __init__(self, ctx, json_nodes):
    self._node = []
    for node in json_nodes['nodes']:
        self._node.append(Node(ctx, node))

  @property
  def node(self):
    return self._node

class Node(object):
  def __init__(self, ctx, node):

    name = node['name']
    op = node['op']
    device="/job:localhost/replica:0/task:0/device:CPU:0" #TODO, remove job/replica/task
    input= []
    attr={}
    if 'input' in node:
        input= node['input']
    if 'attr' in node:
        attr= node['attr']

    self._name = name
    self._op = op
    self._device = device
    self._input = input
    self._attr = attr

  @property
  def device(self):
    return self._device

  @property
  def attr(self):
    return self._attr

  @property
  def name(self):
    return self._name

  @property
  def op(self):
    return self._op

  @property
  def input(self):
    return self._input
