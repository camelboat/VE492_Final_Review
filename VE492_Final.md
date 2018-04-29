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





### 4. Describe biases and variance issue in learning, and how can we select and validate an appropriate model?

### 5. How to control model complexity in linear and logistic regression?

### 6. Using the Least Square as the objective function, we try to find the best set of parameters; what is the statistical justification if the underlying distribution is Gaussian?

### 7. What does the convexity means in either Least Square-based regression or Likelihood-based estimation?

### 8. Gradient Decent has a number of different implementations, including SMO, stochastic methods, as well as a more aggressive Newton method, what are some of the key issues when using any Gradient-based searching algorithm?

It has a sigmoid shapeIt has a sigmoid shapeIt has a sigmoid shape

Newton Method: use second differential

stochastic methods:It has a sigmoid shape



### 9. What are the five key problems whenever we are talking about modeling(Existence, Uniqueness, Convexity, Complexity, Generalizability)? Why they are so important?



It has a sigmoid shape10. Give a probabilistic interpretation for logistic regression? How is it related to the MLE-based generative methods?