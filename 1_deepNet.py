'''
video 2 (modelling) et 3 (running)
https://www.youtube.com/watch?v=oYbVFhK_olY&t=1029s
et
https://www.youtube.com/watch?v=PwAGxqrXSCs


FEED FORWARD NEURAL NETWORK

input > weight > hidden layer 1 (activation function) > weights
               > hidden layer 2 (activation function) > weights
               > output layer
objectiveF: compare the output to the target > cost function (cross entropy)
optimization function (optimizer) > minimize that cost (AdamOptimizer, SGD(stochastic gradient descent), AdaGrad, .. there are 8 solvers in tensorflow)

backpropagation

FEED FORWARD+BACKPROPAGATION = EPOCH (DO THIS CYCLE 'EPOCH' MANY TIMES, depending in the performance of the computer)
'''
import tensorflow as tf 
# Data MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/chaouachi/TensorFlowTuts/data", one_hot=True) #the second parameter is related to electronic stuffs, only one bit is active
'''
We have 10 classes 0-9
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
1=[0,0,1,0,0,0,0,0,0,0]
...
'''
# define the model
n_nodes_hl1=500 # number of nodes in hidden layer 1
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10 # We have 10 classes 0-9
batch_size=100 # number of features (images)

# placeholder in variables

# height * width
x=tf.placeholder('float',[None, 784]) #input data: 784 (28*28) pixels wide, we transform the 2-dimention matrix of the image to 1-D 284 vector
y=tf.placeholder('float') # LABEL of the data (target)

#define the model or the computation graph of the NN
def neural_network_model(data):
	#define some DICTIONARIES
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
	                'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
	                'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
	                'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
	                'biases':tf.Variable(tf.random_normal([n_classes]))}
	# The model
	# (INPUT_DATA * WEIGHTS) + BIAS 
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1=tf.nn.relu(l1) # Rectified linear: activation function

	l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2=tf.nn.relu(l2) # Rectified linear: activation function

	l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3=tf.nn.relu(l1) # Rectified linear: activation function

	output=tf.matmul(l3,output_layer['weights']) + output_layer['biases'] # no add! 

	return output
#define how to run DATA through the predefined model in the tf session
def train_neural_network(x):
	prediction=neural_network_model(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #objFuncton
	optimizer=tf.train.AdamOptimizer().minimize(cost) #stochastic GD || AdamOptimizer has a parameter: learning_rate=0.001 by default
	hm_epochs=10 # how many epochs (cycles ffnn +bbnn)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#start training
		for epoch in range(hm_epochs):
			epoch_loss=0 # calculer la perte pour chaque cycle
			nb_samples=int(mnist.train.num_examples/batch_size) #number of samples
			for _ in range(nb_samples):
				# _ est un indice we don't care about
				epoch_x,epoch_y=mnist.train.next_batch(batch_size) #chunks through the dataset.. for other datasets?? do it yourself :p 
				_,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y}) # c is cost
				epoch_loss+=c
			print('Epoch ',epoch, 'completed out of ', hm_epochs, 'loss=',epoch_loss)
		# statistics
		correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) #prediction and y are equal or not ?
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
# TRAIN THE NETWORK
train_neural_network(x)

