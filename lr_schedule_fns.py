learning_rate = 0.01
lr_param = 0.97
def decay_after_constant(epochs):
    denominator = max(epochs, lr_param)
    return float(learning_rate) * lr_param / denominator

def decay_scaling_factor(epochs):
    return float(learning_rate) * (lr_param ** epochs)

def decay_constant(epochs):
    denominator = 1 + (epochs / lr_param)
    return float(learning_rate) / denominator

def get_learning_rate():
    return learning_rate

def get_lr_param():
    return lr_param
