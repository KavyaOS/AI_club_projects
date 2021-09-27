import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ops.gradients_impl import gradients

# Define some variables

print("--SIMPLE GRADIENT TAPE EXAMPLE--")
x = tf.Variable([-4, -3, -2, -1, 0, 1, 2, 3, 4])
print("x= ", x)

# tf.GradientTape allows us to track our arithmetic operations, allowing
# us to calculate gradients (derivatives) with very few lines of code.

# Everything inside this with statement is tracked by GradientTape.
with tf.GradientTape() as tape:
    y = x**2

print("y = x^2 =", y)

# Getting derivatives of y with respect to x

dy_dx = tape.gradient(y, x)
print("dy/dx =", dy_dx)

print("--Machine Learning--")

# Here we create an example dataset for values of x and y such that
# y = 4*x + 5
#Create an array with numbers 0-1 with interval of 0.01
x = tf.Variable([i*0.01 for i in range(100)], dtype = 'float32')
y = 4*x + 5

print("The first 10 elements of x: ", x[:10])
print("The first 10 elements of y: ", y[:10])

# We will use machine learning to find the best values for 'w' and 'b' in
# the formula "y = w*x + b".
# That is, without telling the computer that y = 4*x + 5, it will figure
# that out through machine learning. By the end, we hopefully get values
# for 'w' and 'b' which are close to '4' and '5'.

#Make our initial guesses for w and b
w = tf.Variable(1.0)
b = tf.Variable(1.)

# To pick the best value for w and b
def call(x, w, b):
    return w*x + b

def MSE(y, y_hat):
    return tf.reduce_mean((y-y_hat)**2)

for i in range(1000):
    with GradientTape() as tape:
        y_hat = call(x, w, b)
        loss = MSE(y, y_hat)

    #Get our derivatives
    gradients = tape.gradient(loss, [w,b])

    # Unpack gradients into separate variables
    # dw represents the derivative of loss with respect to w
    # db represents the derivative of loss with respect to b
    dw = gradients[0]
    db = gradients[1]

    # Update Parameters
    # We multiply by a small number because that ensures that we don't explode w and b
    learning_rate = 0.1

    w.assign(w - dw * learning_rate)
    b.assign(b - db * learning_rate)

print("Through Machine Learning Computer found w to be", w)
print("Through Machine Learning Computer found b to be", b)