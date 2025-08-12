# List characteristics of PyPhenom Attributes
import PyPhenom as ppi
print(dir(ppi))
'''for attr in dir(ppi):
    item = getattr(ppi, attr)
    print(f"\n--- {attr} ---")
    print(dir(item))
print('heavens')
print(dir("Spectroscopy"))'''
spectroscopy = getattr(ppi, 'Spectroscopy', None)
for sub_attr in dir(spectroscopy):
    print(sub_attr)
test = getattr(spectroscopy, 'Composition', None)
print('hetad')
for sub_attr in dir(test):
    print(sub_attr)