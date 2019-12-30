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
<center>
<img src="{{site.baseurl}}/assets/images/sigmoid.jpg" width="75%" height="75%">
</center>

Because in that range we get a variety of output whereas in case our output is greater than 5 the output merely changes. This process is called normalising. Let’s see how our weight initialisation helps us in achieving this.<br>
If we have 5 nodes in the previous layer the weights will be sampled from a uniform distribution from [-limit, limit] where limit is sqrt(6 / (5 + 1)) = 1
Thus our weights will lie in range [-1, 1]. Say the weights are [0.1, -0.2, -0.8, 0.3, 0.3] and the output of the previous node be 0.5 so our summation will be -0.15.<br>
<b>What if we have 100 nodes in the previous layer?</b><br>

In this case the limit will further shrink to accommodate the increase in number of previous layer nodes. Now the limit is [-0.244, 0.244].<br>
Till this point we have randomly initialised the weights of our neural network. Now let’s talk about forward pass.<br>
We’ll describe it via an example it will make the explanation much more simpler.<br>

<center>
<img src="{{site.baseurl}}/assets/images/two_layered.jpg" width="75%" height="75%">
</center>

Consider the above small neural network consisting of only two layers, there is no hidden layer. The first layer is termed as the input layer, no processing happens at this layer whatever is input is outputted the same. The node can be represented in the following way<br>
<center>
<img src="{{site.baseurl}}/assets/images/weights.jpg" width="75%" height="75%">
</center>

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

<center>
<img src="{{site.baseurl}}/assets/images/back.jpg" width="70%" height="70%">
</center>

In the above figure it can be clearly seen that the w1,1 is more responsible for the error because it has greater weight than w2,1. Error responsible is computed by taking the weighted average. e1 = (w1,1 / (w1,1 + w2,1)) * output_error, similarly e2 = (w2,1 / (w1,1 + w2,1)) * output_error.<br>
Let’s see how will we work out when we have three layers.<br>

<center>
<img src="{{site.baseurl}}/assets/images/error.jpg" width="70%" height="70%">
</center>

Well we knew the error produced in the output because we had the correct label for each output node and we only need to find the squared sum of the errors. But <b>how will we find the error in the hidden layer?</b> Because clearly we do not have any explicit data that present in the training other than the labels. So the question is can we get the error produced by the hidden layer in some other way?

<center>
<img src="{{site.baseurl}}/assets/images/error2.jpg" width="70%" height="70%">
</center>

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
<center>
<img src="{{site.baseurl}}/assets/images/eprop.jpg" width="70%" height="70%">
</center>

This was all about which weight to blame more and which to blame less and how to compute the error for the hidden layer indirectly. The main question still remains unanswered that is how to actually update the weight so as to minimise the error.<br>
As you might know that the error that is computed in the end is a function of weights present in the network. If we have only one weight then the plot of error w.r.t the weight might look something as given in the figure.
<center>
<img src="{{site.baseurl}}/assets/images/gd.jpg" width="75%" height="75%">
</center>

If our error depends on two weights than the plot would look something as given in the following figure.
<center>
<img src="{{site.baseurl}}/assets/images/gd2.jpg" width="75%" height="75%">
</center>

Alright, a complex neural network has a way too complicated high dimensional error surface that we can’t visualise. Our intuition is to minimise these errors so we need to pick up the required weights to achieve that.
Why not use brute force? That is to try each and every combination of the weights and look for the error surface whether it reaches the minimum or not. This sounds pretty intuitive to do but is highly computationally expensive and would take years to converge. So we have to come up for a better solution and that’s where mathematics comes to the rescue. We can use what is called <b>gradient descent</b> for our purpose.

### Gradient Descent
Gradient descent is a process which deals with finding the minima in the error surfaces. These can be local as well as global but we always try to search for the global ones let’s save that for later.
Our current task is to find the weights (parameters) that minimises the error. The idea lies in the gradient of the error w.r.t the parameters. Let’s understand the process for a single weight given in fig 8.1. Let’s say we initialised our weight to some random value and it gave us some error. How do we tune the weight? Whether we should add something to the weight or should subtract something? We first find the slope of the surface at that point where we currently are, this can be done by taking a partially derivative of the surface w.r.t. the weight w. The slope gives us the direction in which the graph value will increase if we move in that direction. But as we want to reduce the value(error) we need to move in the opposite direction. So in the figure we start reducing the weight w and we get a lower weight on each iteration.

### Training
As you might have guessed our whole task is to tune the weights in such a way that minimises the error.
In the end all we care about finding is
<center>
<img src="{{site.baseurl}}/assets/images/eq1.jpg" width="25%" height="25%">
</center>

That is how the error E changes when wj,k is tuned and we want to tune it in such a way that it actually minimises the E.
<center>
<img src="{{site.baseurl}}/assets/images/error3.jpg" height="75%" width="75%">
</center>

Well first talk about what the actual error E is? E can be thought of as the sum of squared difference of the actual and the predicted output. We can write it in mathematical form as follows
<center>
<img src="{{site.baseurl}}/assets/images/eq2.jpg" width="50%" height="50%">
</center>

As there can be n outputs we have to take the summation over all of them. One thing we should notice here is that we are trying to differentiate the E w.r.t to wj,k where wj,k implies the link coming from jth node to the kth node. It is pretty clear the on not equal to j will not depend on the wj,k whatsoever. Thus all those terms where on not equal to j will get zero and we’ll only be left with the following equation.
<center>
<img src="{{site.baseurl}}/assets/images/eq3.jpg" width="50%" height="50%">
</center>

It is clear that tk is a constant so it will be zero when we take the differentiation inside. Ok depends on wj,k but the question is how? Ok can be written as sigmoid(netk) and netk can be written as Σj oj * wj,k  where oj  is the output of the previous layer. If we follow the chain rule we’ll get something as follows
<center>
<img src="{{site.baseurl}}/assets/images/eq4.jpg" width="50%" height="50%">
</center>

We can also write the above equation as follows
<center>
<img src="{{site.baseurl}}/assets/images/eq5.jpg" width="70%" height="70%">
</center>

We’ve just dropped the constant 2. We’ve color coded certain things in the above equation, let's see what do these mean? The purple term actually represents the error corresponding to output layer, sometimes also referred as 𝛿 but as we have a more complex nodes, 𝛿 is (tk - ok) multiplied by a squashing function which is given in red. Now the question is how much does it needs to flow in order to update the weights is dependent on the subsequent terms. (Try to link the analogy that we’ve learnt in the above section that complete error is not used to update but a fraction of it). The red term is basically the differentiation of the output of the current node and the green term is the output of the previous layer node.<br>
<b>The whole purple and red term together is 𝛿.</b><br>
The above expression that we derived is only used to update the weights present in between the hidden and output layer. We also need an expression for hidden layer. Let’s derive that too.<br>
As we know that the training example do not provide us the target values for the hidden nodes we actually compute the error terms by summing  the error terms 𝛿k for each next layer node influenced by the current node by weighing each of the 𝛿k by wj,k i.e the weight from the current hidden layer to the next layer. The weight actually characterise the degree to which the hidden unit is responsible for the error.<br>
Let’s start fresh for this.<br>
Before diving into it let’s define some notations.<br>
Wi,j = connection from the ith node to the jth node.<br>
netj = weighted summation of the input converging toward the jth node<br>
oi = is the output of the ith node<br>
𝛿k = is basically an error term used for shortcut to represent some term<br>
Downstream = basically consists of all the links that are connected to the current node to all those nodes present in the next layer<br>
η = is the learning rate<br>
<center>
<img src="{{site.baseurl}}/assets/images/eq6.jpg" width="50%" height="50%">
</center>

Initially we’ve a general term written in blue that we want in the end. We want to know how our error E changes w.r.t wi,j. One thing that we notice is that wi,j can affect the whole network via netj. Thus we’ve applied a chain rule to take that into incorporation. netj on the other hand depends on the wi,j through oi. <br>
Now we can divide the dependence of E on netj in two parts<br>
<ol>
<li>When the jth node is the output node</li>
<li>When the jth node is the hidden node</li>
</ol>

#### Training for output unit weights
<center>
<img src="{{site.baseurl}}/assets/images/eq7.jpg" width="50%" height="50%">
</center>

As you can see that netj can affect the network through the oj we have again incorporated that part. But surprisingly the error is a function of oj and oj, on the other hand can be written as a function of netj (sigmoid dependency).<br>
We call the whole expression as -𝛿j and our updation rule is 
<center>
Δwi,j =  η * 𝛿j * oi
wi,j = wi,j + Δwi,j
</center>

#### Training for hidden unit weights
<center>
<img src="{{site.baseurl}}/assets/images/eq8.jpg" width="50%" height="50%">
</center>

In case of hidden units, wi,j can affect the whole network through netj that we already learnt but as this is a hidden unit the error will be propagated to all those nodes in the next layer that are connected to this node via some link wj,k. Thus we try to take into account those links as well. We represent all those links by downstream. The next layer nodes netk will be affected by the netj via oj thus we have taken that into our chain rule.<br>
Rearranging the term we get the updation rule as for the hidden unit<br>
<center>
Δwi,j = η * 𝛿j * oi
wi,j = wi,j + Δwi,j
</center>

Both the updation rule looks similar but the difference lies in the 𝛿s.

