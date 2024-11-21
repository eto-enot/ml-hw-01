from preprocessing import *
import pickle

class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name):
        if module.startswith('service.'):
            module = module[8:]
        return super().find_class(module, name)
    
def save_model(file_name, obj):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        return _CustomUnpickler(file).load()
