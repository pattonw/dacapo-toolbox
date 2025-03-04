import attr
from cattr import Converter

def serde_test(config):
    # Serialize and deserialize the config
    converter = Converter()
    config_dict = converter.unstructure(config)
    config_copy = converter.structure(config_dict, config.__class__)
    assert config == config_copy
