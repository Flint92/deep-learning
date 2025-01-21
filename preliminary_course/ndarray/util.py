def describe(tensor_x):
    print("x: {}".format(tensor_x))
    print("Shape: {}".format(tensor_x.shape))
    print("Size: {}".format(tensor_x.numel()))
    print("Type: {}\n".format(tensor_x.type()))
