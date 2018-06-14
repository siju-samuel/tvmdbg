# coding: utf-8
"""Tensor class defenition for outputs."""
import json
import base64

class TensorDef(object):
    """Tensor object, which is used for the data transfer of outputs buffer from TVM and CLI.
       This consists of the dtype ,shape and the tensor values"""


    def __init__(self, dtype=None, shape=None, content=None):
        """Initialization function for tensors"""
        super(TensorDef, self).__init__()
        self._dtype = dtype
        self._shape = shape
        self._content = content


    @property
    def dtype(self):
        """Returns the datatype of the tensor"""
        return self._dtype


    @property
    def shape(self):
        """Returns the shape of the tensor"""
        return self._shape


    @property
    def content(self):
        """Returns the content of the tensor"""
        return self._content


    def get_json_string(self):
        """Convert the class to json dictionary and return"""
        values = {'dtype':self._dtype,
                  'shape':self._shape,
                  'content':base64.encodestring(self._content).decode('utf-8')}
        return json.dumps(values)


    def parse_json_string(self, json_str):
        """Parse the json content and returns as a class"""
        values = json.loads(json_str)
        self._dtype = values['dtype']
        self._shape = values['shape']
        self._content = base64.decodestring(values['content'].encode('utf-8'))
