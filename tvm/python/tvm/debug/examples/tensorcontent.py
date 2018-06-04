import json
import base64

class TensorContent(object):
  def __init__(self, dtype=None, shape=None, content=None):
    super(TensorContent, self).__init__()
    self._dtype = dtype
    self._shape = shape
    self._content = content

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  @property
  def content(self):
    return self._content

  def getJsonString(self):
    values = {'dtype':self._dtype, 'shape':self._shape, 'content':base64.encodestring(self._content).decode('utf-8')}
    return json.dumps(values)

  def parseJsonString(self, json_str):
    values = json.loads(json_str)
    self._dtype = values['dtype']
    self._shape = values['shape']
    self._content = base64.decodestring(values['content'].encode('utf-8'))