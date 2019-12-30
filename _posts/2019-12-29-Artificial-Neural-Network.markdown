---
layout : post
title : "Artificial Neural Network"
date : 2019-12-29 08:43:55
---
Before moving on directly to the explanation of what a neural network is we should consider why do we need it. We have learnt about the decision trees, there working principle and their advantages and disadvantages. 
Decision trees are set of if’s and else's, they struggle to find some easy non linear dependencies for e.g. xor. Though we’ve gone through how we can transform a decision tree to incorporate the continuous features but they seem not to work very well because they were designed to work best for the discrete values. ANNs are something that tries to resolve the problem with the decision tree i guess.
Why do we call it Neural Network?
Well there's a history behind why it is called so and we don’t want to go in detail. The structure of an ANN resembles connection in the human brain. We have approximately billion of neurons that are interconnected to each other that helps us in deciding some of the complex tasks for e.g. running, making decisions, driving a car, recognizing faces, etc. 
<img src="{{site.baseurl}}/assets/images/ann.jpg">
Each circle represented in the above figure is called a node (neuron) which is connected to each and every node present in the previous and the next layer thus forming a dense connection.<br>
Layers can be divided into three parts:<br>
<ul>
    <li>Input layer</li>
    <li>Hidden layer</li>
    <li>Output layer</li>
</ul>
Input layer is the first layer that takes the signals from the outside world. Its task is to simply pass whatever is fed to it i.e. no processing at all.<br>

Output layer is the last layer in the neural network and its task is to give some meaningful results for e.g. if we want to decide whether we need to go out for a movie, it should tell us yes or no or say to classify whether the image contains a cat or not.<br>
Hidden layer got their names because whatever they output is not meaningful to us but is meaningful to the layers subsequent to it. So any representation that they have learnt is hidden from us or is so complex that it makes no sense to us, that’s why hidden layer.<br>

# Biological Neuron
<img src="{{site.baseurl}}/assets/images/biological_neuron.jpg">
<b>How does a biological neuron process information?</b>
It gets the stimulus from other neurons via dendrites, all the stimulus (x1, x2, …. xn) are then summed over and checked whether it crosses a certain threshold. If yes then that particular neuron fires. When it fires it tells other neurons connected to it that hey I think I’ve got some useful information that you might find helpful.
The above figure represents a single node (neuron). The whole process can be divided into two parts. The first part is the accumulation of the weighted signals. Weights let us know about how important the signal is, they might amplify or deamplify it. The net is computed by summing all the weighted signal. The second part is to pass this net to an activation function which tells us whether the given net is useful or not. Say we passed our net = 5 to a step function whose threshold is 0. Thus after passing it to the step activation function our output will be 1. If net is -5 then output will be -1.
<b>Questions</b>
<ol>
    <li>How to choose weights?</li>
    <li>How to choose the activation function?</li>
    <li>How many layers shall we have?</li>
    <li>How many nodes in each layer shall we have?</li>
</ol>

The weights are the key part in neural network, it is this thing that decides which function to learn for e.g. whether to learn a quadratic function or a hyperplane. How to get the right function based on the dataset is what neural network training is all about. Choosing the right activation function is important. For example, if we choose a linear activation function then our network won’t be able to learn non linearity and will fail in creating non linear decision boundary.<br>
To decide how many layers and how many nodes each layer should have, totally depends upon the context. If the function that you are trying to learn is fairly complex we might need several nodes and several layers. There is no perfect or predefined criteria that guides us in choosing the number of layers and nodes. 

## Training a Neural Net
The training process can be divided into two phases:
<ol>
<li>Forward Propagation</li>
<li>Backward Propagation</li>
</ol>

### Forward Propagation
This phase consist of randomly initialising the neural network weights. The question is how random? I mean can it be {8, 9, 100, 10000382, 3, 2} will this suffice. No actually the numbers are random but the interval from which we sample depends on the two layers which is connected via these weights. For example keras uses glorot_uniform as its initialiser.<br>
It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of units in the previous layer and fan_out is the number of units in the current layer.
Alright but why?<br>
Say you have 5 units in the previous layer and as you know neural network are densely connected so each node in the current layer will be connected to the previous 5 nodes. Assuming those 5 nodes output comes from a sigmoid activation thus there range will be (0, 1). If we initialised our weights to be say {3, 5, 10, 7, 20} then the summation to the current node from the previous node will be “3 * (0, 1) + 5 * (0, 1) + 10 * (0, 1) + 7 * (0, 1) + 20 * (0, 1)” if we say that all the previous nodes output is 0.5 then our summation will be 45 * 0.5 = 22.5 if this is passed to a sigmoid activation function then it will saturate the sigmoid which means the output will be close to 1.<br>
Let me give you an example
σ(5) = 0.99330714907<br>
σ(10) = 0.99995460213<br>
σ(15) = 0.99999969409<br>
σ(20) = 0.99999999793<br>
σ(25) = 0.99999999998<br>
As you can see even though our net varies from 5 to 25 the output it produces changes a little. Well we want if the net differs drastically so does the output then only our model will be able to distinguish. One way to deal with this kind of problem is to limit the net to be in the range of [-1, 1]
<img src="{{site.baseurl}}/assets/images/sigmoid.jpg">
Because in that range we get a variety of output whereas in case our output is greater than 5 the output merely changes. This process is called normalising. Let’s see how our weight initialisation helps us in achieving this.<br>
If we have 5 nodes in the previous layer the weights will be sampled from a uniform distribution from [-limit, limit] where limit is sqrt(6 / (5 + 1)) = 1
Thus our weights will lie in range [-1, 1]. Say the weights are [0.1, -0.2, -0.8, 0.3, 0.3] and the output of the previous node be 0.5 so our summation will be -0.15.<br>
<b>What if we have 100 nodes in the previous layer?</b><br>
In this case the limit will further shrink to accommodate the increase in number of previous layer nodes. Now the limit is [-0.244, 0.244].<br>
Till this point we have randomly initialised the weights of our neural network. Now let’s talk about forward pass.<br>
We’ll describe it via an example it will make the explanation much more simpler.<br>
<img src="{{site.baseurl}}/assets/images/two_layered.jpg">
Consider the above small neural network consisting of only two layers, there is no hidden layer. The first layer is termed as the input layer, no processing happens at this layer whatever is input is outputted the same. The node can be represented in the following way<br>
<img src="{{site.baseurl}}/assets/images/weights.jpg">
The left part is called an aggregator or summer as it performs the weighted summation over all the inputs. The right part task is to apply the transfer function to the weighted sum and hand over the output to the next layer nodes.<br>
Let me first walk you through the notation wi,j represents the weight of the connection that connects the ith node to the jth node in the next layer.<br>
<center>
layer_2_net_1 = inputs_1 * w<sub>1,1</sub> + inputs_2 * w<sub>2,1</sub><br>
layer_2_net_2 = inputs_1 * w<sub>1,2</sub> + inputs_2 * w<sub>2,2</sub><br>
outputs_1 = sigmoid(layer_2_net_1)
outputs_2 = sigmoid(layer_2_net_2)
</center>

### Backpropagation
Now what, after getting the outputs we need to check whether these outputs matches the desired outputs. If not by how much factor are we wrong? And how do we minimise these errors. <br>
By what we have learnt above we can be sure that we need to tune the weights to reduce the errors. The question is how? One intuition can be we can blame the previous link for the errors that we got in the current layer but the question is how much should we blame the links? Should we blame equally? Or there has to be a some part?<br>
Let’s discuss this idea a little bit more. Say in a firm a project failed to meet the requirement, it was known that the project was being led by two people well those two people will be more responsible than the rest of the members associated with the projects. The same analogy is applied here the connections having more weight are more responsible for the error produced.<br>
<img src="{{site.baseurl}}/assets/image/back_prop.jpg">
In the above figure it can be clearly seen that the w1,1 is more responsible for the error because it has greater weight than w2,1. Error responsible is computed by taking the weighted average. e1 = (w1,1 / (w1,1 + w2,1)) * output_error, similarly e2 = (w2,1 / (w1,1 + w2,1)) * output_error.<br>
Let’s see how will we work out when we have three layers.<br>
<img src="{{site.baseurl}}/assets/image/error.jpg">
Well we knew the error produced in the output because we had the correct label for each output node and we only need to find the squared sum of the errors. But <b>how will we find the error in the hidden layer?</b> Because clearly we do not have any explicit data that present in the training other than the labels. So the question is can we get the error produced by the hidden layer in some other way?
<img src="{{site.baseurl}}/assets/image/error2.jpg">
Error produced by the hidden layer can be indirectly computed via the error produced by the subsequent layers. Say we have a three layer network and we know the error produced at the output layer let’s call it eoutput,1 and eoutput,2. We can take use of these errors to compute the error produced by the hidden layers. It is similar to blaming the weights for the error they produced.<br>
Let’s look at the hidden layer node 1, the error produced by that node must have propagated through the two links via w1,1 and w1,2.<br>
How much of eoutput,1 and eoutput,2 is present in ehidden,1. This is a little tricky and can get a bit confusing the first time.<br>
eoutput,1 is the error that was produced by w1,1 and w2,1 so how much of the error corresponds to w1,1?  eoutput,1 * (w1,1 / (w1,1 + w2,3) ), similarly 
eoutput,2 is the error that was produced by w1,2 and w2,2 so how much of the error corresponds to w1,2?  eoutput,2 * (w1,2 / (w1,1 + w2,3) )
<br>
Thus using the above analogy the error produced by the ehidden,1 is equal to the error produced at w1,1 and w1,2.
<center>
ehidden,1 = eoutput,1 * (w1,1 / (w1,1 + w2,3) ) + eoutput,2 * (w1,2 / (w1,1 + w2,3) )
</center>
You can refer to the figure 7.4 given below to get a better understanding of what is happening.
<img src="{{site.baseurl}}/assets/image/eprop.jpg">

