from dacapo_toolbox.converter import converter


def serde_test(config):
    # Serialize and deserialize the config
    config_dict = converter.unstructure(config)
    config_copy = converter.structure(config_dict, config.__class__)
    assert config == config_copy, (config, config_copy)
