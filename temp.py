import tensorflow as tf

def s_mat_mul(x,y):
    return  tf.matmul(x,y)

def s_mat_add(x,y):
    return tf.add(x,y)

@tf.function
def simple_test(x,y)
