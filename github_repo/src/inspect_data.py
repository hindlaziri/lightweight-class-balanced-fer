from pathlib import Path
import pickle

path = Path('/home/ubuntu/telkomnika_fer/data_train.pt')
with path.open('rb') as handle:
    obj = pickle.load(handle)
print(type(obj).__name__)
try:
    print('length', len(obj))
except TypeError:
    print('length unavailable')
if isinstance(obj, list):
    for i, item in enumerate(obj[:3]):
        print('item', i, type(item).__name__)
        if isinstance(item, dict):
            print('keys', sorted(item.keys()))
            for key, value in item.items():
                print(key, type(value).__name__, getattr(value, 'shape', None), str(value)[:120])
        elif isinstance(item, (tuple, list)):
            print('tuple length', len(item))
            for j, value in enumerate(item):
                print('tuple item', j, type(value).__name__, getattr(value, 'shape', None), str(value)[:120])
else:
    print('repr', repr(obj)[:1000])
