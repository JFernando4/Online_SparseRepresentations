
class Config:
    """
    Used to store the arguments of all the functions in the package Experiment_Engine. If a function requires arguments,
    it's definition will be: func(config), where config has all the parameters needed for the function.
    """
    def __init__(self):
        pass


def check_attribute_else_default(object_type, attr_name, default_value, choices=None):
    if not hasattr(object_type, attr_name):
        print("Creating attribute", attr_name)
        setattr(object_type, attr_name, default_value)
    if choices:
        if getattr(object_type, attr_name) not in choices:
            raise ValueError("The possible values for this attribute are: " + str(choices))
    return getattr(object_type, attr_name)


def check_dict_else_default(dict_type, key_name, default_value):
    assert isinstance(dict_type, dict)
    if key_name not in dict_type.keys():
        dict_type[key_name] = default_value
    return dict_type[key_name]
