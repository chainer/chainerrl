from chainer import functions as F

def clipped_loss(x, t):
    diff = x - t
    abs_loss = abs(diff)
    squared_loss = diff ** 2
    abs_loss = F.expand_dims(abs_loss, 1)
    squared_loss = F.expand_dims(squared_loss, 1)
    return 0.5 * F.sum(F.min(F.concat((abs_loss, squared_loss), axis=1), axis=1))
