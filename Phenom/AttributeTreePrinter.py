# Attribute Tree Printer
import PyPhenom as ppi
import types

def print_attr_tree(obj, name='root', depth=0, max_depth=5, visited=None):
    if visited is None:
        visited = set()

    indent = '    ' * depth
    obj_id = id(obj)
    
    if obj_id in visited:
        print(f"{indent}{name}: (Already visited)")
        return
    visited.add(obj_id)

    print(f"{indent}{name}: {type(obj).__name__}")

    if depth >= max_depth:
        print(f"{indent}    ... (max depth reached)")
        return

    for attr_name in dir(obj):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue  # Skip dunder methods

        try:
            attr_value = getattr(obj, attr_name)
        except Exception:
            continue  # Skip if inaccessible

        if isinstance(attr_value, (types.ModuleType, types.FunctionType, type)):
            # Don't go into modules or functions or classes
            print(f"{indent}    {attr_name}: {type(attr_value).__name__}")
        else:
            print_attr_tree(attr_value, attr_name, depth + 1, max_depth, visited)

# Run it on PyPhenom.Spectroscopy
# print_attr_tree(ppi.Spectroscopy, 'Spectroscopy', max_depth=4)
# print_attr_tree(ppi.Spectroscopy.Spectrum, 'Spectrum', max_depth=4)
# print_attr_tree(ppi.Spectroscopy.EdxAcquisition, 'EdxAcquisition', max_depth=4)
