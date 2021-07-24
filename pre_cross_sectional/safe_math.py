import numpy.ma.core as c


banned_command = ['exit', 'input', 'exec', 'print', 'eval', 'compile', 'locals', 'globals',
                   'help', 'setattr', 'all', 'dir', 'next', 'slice', 'any', 'id',
                  'object', 'ascii', 'enumerate', 'staticmethod', 'open', 'isinstance', 'ord',
                  'bytearray', 'issubclass', 'super', 'bytes', 'iter', 'callable', 'property',
                  'format', 'chr', 'frozenset', 'vars', 'classmethod', 'getattr', 'repr', '__import__',
                  'reversed', 'map', 'hasattr', 'delattr', 'hash', 'memoryview']
banned_command_dict = {x: None for x in banned_command}
allowed_math_dict = {x: c.__dict__[x] for x in c.__all__}

def safe_eval(string: str, variable=None):
    return eval(string, {}, banned_command_dict | allowed_math_dict | {'x': variable})
