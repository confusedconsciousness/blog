---
layout : post
title : "Artificial Neural Network"
date : 2019-12-29 08:43:55
---
Before moving on directly to the explanation of what a neural network is we should consider why do we need it. We have learnt about the decision trees, there working principle and their advantages and disadvantages. 
Decision trees are set of if’s and else's, they struggle to find some easy non linear dependencies for e.g. xor. Though we’ve gone through how we can transform a decision tree to incorporate the continuous features but they seem not to work very well because they were designed to work best for the discrete values. ANNs are something that tries to resolve the problem with the decision tree i guess.
Why do we call it Neural Network?
Well there's a history behind why it is called so and we don’t want to go in detail. The structure of an ANN resembles connection in the human brain. We have approximately billion of neurons that are interconnected to each other that helps us in deciding some of the complex tasks for e.g. running, making decisions, driving a car, recognizing faces, etc. 
![Neural Network](/assets/images/ann.jpg)
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
