# video 2: https://www.youtube.com/watch?v=oYbVFhK_olY&t=1029s
import tensorflow as tf 

# shunk1: the model
#constants
x1=tf.constant(5)
x2=tf.constant(6)
results=tf.multiply(x1,x2)
print(results)

#Shunk2: RUN it
with tf.Session() as sess:
	output=sess.run(results)
	print(output)
print(output)
# sess.close() #this is automaticaly done with WITH clause