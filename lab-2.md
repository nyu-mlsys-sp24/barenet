# Lab-2: Implementing Multilayer Perceptron (MLP)

The goal of this lab is demonstrate how to build a simple MLP model using the basic tensor operations developed in Lab-1.
Alas, instead of autograd, we'll perform manual gradient calculation.
Nevertheless, this Lab tries to structure the code to be modular and readable.

## Preliminary: Obtaining the lab code
For this Lab, you will work in the same respository that you've worked on for Lab-1. The compilation procedure is the same 
as that for [Lab-1](https://github.com/nyu-mlsys-sp24/barenet/blob/master/lab-1.md#compilation).

## Lab-2 Instructions

### Understanding the MLP training loop 

Before starting coding for Lab-2, it is important to first read the code of `train_mlp.cu` to understand the basic MLP program structure.  
We can see that its main functionality is a training loop implemented by the [`train` function](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L25).  

The `train` function first loads the MNIST training and test dataset [here](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L27).  Then, it constructs the MLP model by calling the MLP object constructor with hyperparameters such as batch size, the input feature dimension size and a vector of dimension sizes representing each layer's out dimension size. By [default](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L84), the model has two layers, with the first layer's weight matrix 784x16 (input dimension:784, hidden dimension:16) and the second layer's weight matrix 16x10 (hidden dimension:16, output dimension:10).  The input dimension is 784 because each MNIST image has 28 by 28 pixel of floats. The output dimension is 10 because we are trying to classifying 10 digits (0-9). You can change the number of layers, batch size, and the hidden dimension using [commandline argumnents](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L91).  After the MLP is constructed, we initialize its weights with `mlp.init()`.  We then constructs the SGD optimizer with learning rate 0.01 [here](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L41).

After the model and the optimizer are constructed, the training can begin. It goes through several epochs. In each epoch, it goes through the entire training data set sequentially one batch at a time.  For each batch, we call `model.forward(input_images, logits)` to perform the forward computation on a batch of `input_images` and put the resulting logit values in the output tensor `logits`.  Then, we calculate the CrossEntropy loss for the batch of data using operator [`op_cross_entropy_loss`](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L66C1-L67C1). The operator `op_cross_entropy_loss` is a manually fused operator that calculates the loss given the logits tensor (computed by `model.forward`), the `targets` tensor containing the batch's training labels.  In addition, the operator also calculates the gradients of the logits and put them in the `d_logits` tensor. With the `d_logits` gradients, we can start the rest of the backward computation by calling `model.backward(input_images, d_logits, d_input_images).`.

### Step-1: implementing `op_cross_entropy_loss`


### Step-2: implementing the forward computation of MLP

### Step-3: implementing the backward computation of MLP


## Training MLP

Correctness will constitute 70% of Lab-1's score. 
We will evaluate the correctness of your Lab-1 using a simple unit test file `test.cu`.  You need a GPU in order to run the test! 
Once you finish compilation, 
run the unit test as follows:

```
$ ./test
```

If your passed the unit test, the output will look like this:
```
$./test
slice passed...
op_add passed...
op_multiply passed...
op_sgd passed...
matmul passed...
op_sum passed...
All tests completed successfully!
```

## Performance
Performance will constitute 30% of Lab-1's score. We will compare your kernel's performance with those of the instructor's 
own basic implementation.

## Lab-1 instructions

In Lab-1, you will complete the necessary code in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh` and pass 
simple unit tests. Your implementation should also be performant in order to get good performance scores.

You should first carefully read through the code in `utils/tensor.cuh`. This file defines our tensor abstraction. 
Here are the things you should know about our tensors:

- The tensor is templated so that we can have different data types for the elements stored in it.

- The tensor's internal buffer is ref-counted using C++ smart pointer and its corresponding memory storage is automatically freed when there are no more references to it.
In other words, you do not need to worry about needing to free the memory.

- For our labs, the tensor is *always* 2-dimensional, with the first dimension named `h` (height) and second dimension named `w` (width).
A row vector has `h=1` and a column vector has `w=1`.

- The Macro `Index(t, i,j)` will  be handy for accessing the element at [i,j] coordinate of tensor t. The Macro `IndexOutOfBound(t, i, j)` will be handy for testing whether [i,j] is out of bound for tensor t.


Next, complete the necessary functions  in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh`, in the given order.
Read through the code skeleton, and fill in your code whenever you see a comment that says `Lab-1: please add your code here`.
After finishing each file, you should be able to pass a portion of the unit test. Debug and complete that portion of the unit test 
before moving on to the next lab file.

## Saving your progress

You want to save whatever progress you've made on the lab and back it up frequently so that losing your laptop does not result in the loss of your lab work.  To do so, you commit your file modifications so far and push those commits (aka back them up) to your remote respository on Github.  You do so by typing the following:
```
$ git commit -am "Some meaningful commit message"
$ git push origin master
```

Do the above frequently while you work on the lab.  However, it is generally frowned upon to commit a change that does not compile.

## Hand-in procedure

To hand in your lab, first commit all of your modifications by following the instructions in the section on [Saving your progress](#Saving-your-progress). Second, make a tag to mark the latest commit point as your submittion for Lab1. Do so by typing the following:
```
$ git tag -a lab1 -m "submit lab1"
```

Finally, push your commit and your tag to Github by typing the following
```
$ git push --tags origin master
```

You should double check that your commit and your tag is correctly pushed to the Github by double checking 
on the github webpage. See the Screenshot below as an example ![](https://news.cs.nyu.edu/~jinyang/GithubScreenshot.jpg)



That's it.  **Please do not delete or modify your tag after the Lab submission date. We'll make a copy of the tagged commit from your Github repository immediately after the Lab submission date has passed.**  After you've tagged, you can continue to commit new changes and push your new commits to Github as you move on to doing Lab-2.


