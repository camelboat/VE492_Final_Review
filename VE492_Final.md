# VE492 Final

## Learning and search methods:

### 1. What is the loss function? Give three examples(Least Square, Logistic, Hinge) and describe their shapes and behaviors;

In mathematical optimization, statistics, ecnometrics, decision theory, machine learning and computational neuroscience, a loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function.

The difference between your model prediction f(x) and real value y

Least Square: $L(Y,f(x))=(Y-f(x))^2$

Logistic Loss:  $V(f(x),y)=\log(1+e^{-z})$	

Hinge: $V(f(x),y)=max(0,1-z)$

Their shape are on page 131



### 2. Using these losses to approach the actual linear boundary, inevitably some risks will be incurred: give two different approaches to remedy the risk using the SVM-based hinge loss as an example;

risk1: it is hard to make sure that the kernel function can linearly classify the samples in eigen-space

risk2: it is hard to determine the likely linear separable is caused by overfitting

method1: use soft margin, that is, allow the SVM to give wrong prediction on some of the samples

method2: add regularization such as L1 and L2 norm



### 3. How many possible models are there given a set of training data? What is the key assumption of PCA learning for model selection?

There are infinite number of possible models given a set of training data.

PCA assume that the main features of data are on orthogonal directions.



### 4. Describe biases and variance issue in learning, and how can we select and validate an appropriate model?

Bias: an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs(underfitting).

Variance: an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs(overfitting)

High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities.

Models with low bias are usually more complex, enabling them to represent the training set more accurately, but it could represent a large noise component in the training set, making their predictions less accurate, despite their added complexity. In contrast, models with higher bias tend to be relatively simple, but may produce lower variance predictions when applied beyond the training set.

One way of resolving this trade-off is to use mixture models and ensemble learning.

ensemble learning: use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.



### 5. How to control model complexity in linear and logistic regression?

Use dimension reduction or feature extraction


### 6. Using the Least Square as the objective function, we try to find the best set of parameters; what is the statistical justification if the underlying distribution is Gaussian?

Bayesian decision theory?

### 7. What does the convexity means in either Least Square-based regression or Likelihood-based estimation?



### 8. Gradient Decent has a number of different implementations, including SMO, stochastic methods, as well as a more aggressive Newton method, what are some of the key issues when using any Gradient-based searching algorithm?



### 9. What are the five key problems whenever we are talking about modeling(Existence, Uniqueness, Convexity, Complexity, Generalizability)? Why they are so important?




###10. Give a probabilistic interpretation for logistic regression? How is it related to the MLE-based generative methods?


###11. Compare the generative and discriminative methods?

### 12. For the regular and multinomial Naive Bayes, what are their key assumptions? Why the multinomial method can be more context sensitive?

### 13. What are the key advantages of linear models? What are the key problems with the complex Neural Network?

### 14. What are three alternatives to approach a constrained maximization problem?

### 15. What is the dual problem? What is strong duality?

### 16. What are the KKT conditions? What is the key implication of them? Including the origin of SV?

### 17. What is the ideal of soft margin SVM, how it is a nice example of regularization?

### 18. The ideal of kernel? Why not much additional computational complexity?

### 19. What is the general idea behind the kernel? What key computation do we perform? Why is it so general in data modeling?

### 20. Why we often want to project a distance "measure" to a different space?

## probabilistic graphical model

### 1. Compare the graphical representation with feature vector-based and kernel-based representations 



### 2. Explain why sometime a marginal distribution has to be computed in a graphical model



### 3. Why a graphical model with latent variables can be a much harder problem?





### 1.

(a) how a margin-based linear classifier like SVM can be even more robust than Logistic regression?

logistic regression is much more sensitive to the margin of different labels, and we may have to consider a distribution or model of the data before we start to train the model, which may bring more risk, hence SVM is more robust

(b)

first use dimension reduction, then use kernel function to control the overlapping boundary



### 2.

(a) convolution-based deep learning doesn't use the apparent feature of an object selected by human. It can extract feature by itself from data, and learn about the object's feature related to the background environment, which is more important than feature itself, and won't be affected by this dilemma.

(b) linear regression may use features that are not orthogonal to each other, are not normalized, and may even have much relationship with others. Unlike convolution-based deep learning, we can't make sure if feature we select for linear regression indeed have relation with their labels, hence results from these features may not been meaningful.



### 3.

(a) kernel is one complex function applied to all data, while neural networks use a certain simple function to  separate those nonlinear data into many small parts.

(b) 奇函数的选择

组合的方式

在考虑数据间关系的同时还要兼顾组合量的关系

既要使神经网络具有足够的表达能力，又要考虑数学计算上的能力（但会损失一定高阶的数学特征）



### 4.

(a) why a gradient-based search is much more favorable than other types of searches?

如果没有梯度的方向给定的确定性和唯一性， you have to 穷举各个方向的参数变化

(b) what would be the possible ramifications of having to impose some kinds of sequentiality in both providing data and observing results?



### 5.

(a) use linear regression as the example to explain why L1 is more aggressive when trying to obtain sparser solutions compared to L2?

要求了解四个范数



(b) 

### 6.

(a) What is the key difference between a supervised vs. unsupervised learnings

when an unlabeled data is presented, it has to been marginally estimated which label it should has. Although you can guess for many times, you can't make sure it is the global solution or it is a local solution.

(b) why unsupervised learning does not guaranty a global solution?

search for mathematical formulas



### 7.

(a) provide a Bayesian perspective about the forwarding message to enhance an inference

$\alpha$ recursion?

(b) how to design a more generalizable HMM which can still converge efficiently?

the first several data should be the strongest features, which can give you the prior to deal with other data.



### 8. 

(a)

depth increase, 



(b)

结果和先验相符



### 9.

(a)

有选择地选择feature, 但又在不是穷举的情况下以保证模型有一定的泛化能力



(b)



### 10.



数学形式+解释



