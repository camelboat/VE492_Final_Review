# VE492 Final

## Learning and search methods:

### 1. What is the loss function? Give three examples(Least Square, Logistic, Hinge) and describe their shapes and behaviors;

In mathematical optimization, statistics, econometrics, decision theory, machine learning and computational neuroscience, a loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function.

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

Use regularization, such as weight decay to modify the training criteria.

Collect more data.

Use validation set to train and select hyper parameters.

Use dimension reduction or feature extraction.




### 6. Using the Least Square as the objective function, we try to find the best set of parameters; what is the statistical justification if the underlying distribution is Gaussian?

By Bayesian statistics theory, if the underlying distribution is Gaussian, we should set our prior probability distribution of these parameters to be Gaussian distributed, then use them to do Bayesian statistics.

Here prior is:

$p(\theta)$



$P(\theta|x)=\frac{P(x_1,x_2...x_m)P(\theta)}{P(x_1, x_2, ..., x_m)}$

in which

$P(x_1, x_2, ...x_m)=\sum_{i=1}^{m}P(\theta_i)P(x|\theta_i)$



### 7. What does the convexity means in either Least Square-based regression or Likelihood-based estimation?

least square-based regression and likelihood-based estimation are two convex optimization problems, which only has one global minimum and the local minimum is same as global minimum. The convexity makes optimization easier than the general case and first-order conditions are sufficient conditions for optimality.



### 8. Gradient Decent has a number of different implementations, including SMO, stochastic methods, as well as a more aggressive Newton method, what are some of the key issues when using any Gradient-based searching algorithm?

1. with the increasing volume of data, requirement of computational capability will increase during gradient calculation
2. When applying gradient-based searching algorithm to a non-convexity problem, it may not be possible for the algorithm to finally find the global minimum/maximum value
3. step modification for each minimum gradient search must be annotated based on trying



### 9. What are the five key problems whenever we are talking about modeling(Existence, Uniqueness, Convexity, Complexity, Generalizability)? Why they are so important?

Existence:  There exist a model that is fit to our task.

Uniqueness: new data can only have one output through our model

Convexity: model has only one global minimums

Complexity: model is complex enough to represent the data with enough low bias

Generalizability:  variance of the model is low enough, it can achieve a good prediction accuracy out of the training data.



###10. Give a probabilistic interpretation for logistic regression? How is it related to the MLE-based generative methods?

For logistic regression, the objective function is:

$y=\frac{1}{1+e^{-(w^Tx+b)}}$

From which we can have:

$\ln{\frac{y}{1-y}}=w^Tx+b$

if we treat y as the probability for x to be a positive observation, then 1-y is negative observation probability, where  y/(1-y) called odds, is the relative probability for x to be positive, and log odds, also called logit, which is $\ln{\frac{y}{1-y}}$

Hence we can see that logistic regression's objective function is trying to use linear regression's model to get the logit of real labels, which is the reason why we call it logistic regression(or logit regression)

Now if we treat y as the posterior $p(y=1|x)$

Then we can have:

$\ln{\frac{p(y=1|x)}{p(y=0|x)}}=w^Tx+b$

hence it is obvious that:

$p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}$

$p(y=0|x)=\frac{1}{1+e^{w^Tx+b}}$

Then use MLE we can estimate value of w and b

For dataset $\{(x_i, y_i)\}_{i=1}^m=$, maximum log likelihood is:

$l(w,b)=\sum_{i=1}^{m}\ln p(y_i|x_i;w,b)$



###11. Compare the generative and discriminative methods?

generative modeling(生成模型): models how the data was generated in order to categorize a signal. It asks the question: based on my generation assumptions, which category is most likely to generate this signal. It generate the joint distribution $P(x,y)$, so it can represent data distribution based on statistics, and represent the similarity of data with same category. However, it does not care about the accurate margin for each categories. Generative modeling has higher convergence rate, and when there are latent variables, we can only use generative modeling, to learn it, nor the discriminative methods.

discriminative modeling(判别模型): does not care about how the data was generated, it simply categorizes a given signal. DM directly learn from data, generate decision function or conditional probability distribution $P(y|x)$ and can not represent the data's own feature. However it tries to find the best discriminative margin plane, and can represent the different between data with different labels. Since it directly face to prediction, it may have higher learning accuracy, and can have abstract extraction from data, and use them as new features, which can simplify the question.

### 12. For the regular and multinomial Naïve Bayes, what are their key assumptions? Why the multinomial method can be more context sensitive?

A Naïve Bayes assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature (attribute conditional independence assumption).

Multinomial Naive Bayes classifier is a specific instance of a Naive Bayes classifier which uses a multinomial distribution for each of the features.

Multinomial Naive Bayes classifier works well for dat which can be easily turned into counts, because here samples represents the frequencies with which certain events have been generated by a multinomial $(p_1, ... p_m)$ and feature vector $X=(x_1,...x_m)$ is then a histogram with $x_i$ counting the number of times event $i$ was observed in a particular instance. Since in context tasks, it is easy to count the occurrence of a word, the multinomial method can be more context sensitive.



### 13. What are the key advantages of linear models? What are the key problems with the complex Neural Network?

Key advantages of linear models:

1. easy to generate the model, has good comprehensibility


1. Can be really accurate when the relationships between the data and labels can be proved to be linear
2. Many linear models may have the closed-form solution, which may cost little training time compared to non-linear models.

Key problems with the complex neural network:

1. needs a lot of data and cases(don't perform well on small data sets, where Bayesian approaches have advantage)
2. trained model is a black box, and it is hard to explain their actual meaning(hard to interpret the model)
3. increase risk of overfitting
4. computation costs

### 14. What are three alternatives to approach a constrained maximization problem?

1. Gradient-based searching approach(directly solve the original objective function)
2. Design a different optimization problem with no constraint, and transform its solution into the solution of the original constrained problem
3. Karush-Kuhn-Tucker methods(KKT method)



### 15. What is the dual problem? What is strong duality?

A dual problem is an optimization problems that may be viewed from either of two perspectives, the primal problem or the dual problem. The solution to the dual problem provides a lower bound to the solution of the primal problem.

Strong duality means the duality gap is zero, which is the difference between the values of any primal solutions and any dual solutions.



### 16. What are the KKT conditions? What is the key implication of them? Including the origin of SVM?

Suppose we want to do solve the SVM like this:

$\text{min}_{w,b}\frac{1}{2}||w||^2$

$\text{s.t. }y_i(w^Tx_i+b)\ge1, i=1,2,...,m$

We can use the Lagrangian multiplier method to find its dual problem, that is, add Lagrange multiplier $\alpha_i\ge0$ to these two functions, which can be turned into:

$L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^{m}\alpha_i(1-y_i(w^Tx_i+b))$

in which $\alpha=(\alpha_1;\alpha_2;...\alpha_m)$, and if we let the partial derivative of $L(w,b,a)$ to w and b to be 0, we can have that:

$w=\sum_{i=1}^{m}\alpha_iy_ix_i$

$0=\sum_{i=1}^{m}\alpha_iy_i$

Hence we can eliminate w and b from $L(w,b,\alpha)$, which is the dual problem:

$\text{max}_{\alpha}\sum_{i=1}^{m}\alpha_i-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j$

$\text{s.t. }\sum_{i=1}^{m}\alpha_iy_i=0, \alpha_i\ge0, i=1,2,...,m$

and then if we solve $\alpha$, we can get the model $f(x)$ which is:

$f(x)=w^Tx+b=\sum_{i=1}^{m}\alpha_iy_ix_i^Tx+b$

The procedure above should follow KKT conditions, which are:

$\alpha_i\ge0$,

$y_if(x_i)-1\ge0$

$\alpha_i(y_if(x_i)-1)=0$

Then, for any observation$(x_i,y_i)$, it will always have $\alpha_i=0$ or $y_if(x_i)=0$

If $\alpha_i=0$, then it has no influence to the model

If $\alpha_i>0$, then $y_if(x_i)=1$, so the corresponding point must be on the maximum margin, which is a  support vector. Hence when the training is finished, most of the training samples don't have to be reserved, since the final model only concerns support vectors.



### 17. What is the idea of soft margin SVM, how it is a nice example of regularization?

In real-world task, it is hard to find a certain kernel function that can make the training samples be available to linear separated in feature space, or is hard to justify whether the seemly linearly separated results is caused by overfitting. To solve this problem, idea of soft margin SVM is introduced, which allows some samples to be predicted wrong on the SVM.

The general form of soft margin SVM can described by:

$\text{min}_{f}\Omega(f)+C\sum_{i=1}^{m}l(f(x_i),y_i)$

in which $\Omega(f)$ is the structural risk to describe some features of model $f$ and the second term is empirical risk which is used to describe the fitting degree. Since $\Omega(f)$ introduces a way to input professional knowledge and people's mind, which is helpful for cutting down the hypothesis space and cut down the overfitting risk, this equation can be called a regularization problem, in which $\Omega(f)$ is the regularization term, and $C$ is the regularization constant. Consequently, it is a nice example of regularization.



### 18. The ideal of kernel? Why not much additional computational complexity?

For the dataset that can't be linearly separated in their feature space, we want to add a kernel function in order to make them linearly separable in a feature space with higher dimensions.

Because as long as we can calculate the inner product in the feature space, we do not need the mapping explicitly, (do not explicitly do $\phi(x)$), there is not much additional computational complexity.



### 19. What is the general idea behind the kernel? What key computation do we perform? Why is it so general in data modeling?

First question same to 18.

The key computation we perform is the inner product $x^Tx$

It is general in data modeling because:

1. It can allow us to use convex optimization tool to solve non-linear model
2. It is much more efficient than directly build $\phi(x)$ and then compute the inner product



### 20. Why we often want to project a distance "measure" to a different space?

Because the data may not be able to separate in the original feature space, we want to project them into a different space where data with different labels have the maximum distance in order to generate our learning models.

Kernelization, to project the original $X^TX$ to a "Feature Space" of higher dimensions("Feature Vectors") in order to better distinguish and measure the difference and similarity between the X and X'.



## probabilistic graphical model

### 1. Compare the graphical representation with feature vector-based and kernel-based representations 

Feature vector: need more training data, smooth margin, can have infinite dimensions

Kernel: project low-dimension data pairs to high-dimensions, output change is not smooth, margin is not stable

Graphical model: use independent and identically independent random variables setting(iid), can use prior and posterior to reduce the model complexity.



### 2. Explain why sometime a marginal distribution has to be computed in a graphical model

Because without the graphical model, there can be too many variables that contributes to that marginal distribution which we can't compute directly. With the graphical model, which is actually a dimension reduction of the actual events, we can eliminate these variables and get the marginal distribution, hence sometime a marginal distribution has to be computed in a graphical model.

Ancestor and parents prior can have influence on the post current condition, however, since prior is a mixture, which have the total influence 1 on the current condition and it is unknown which part has actual influence to the current condition, we have to compute marginal distribution in order to  have the clear discrimination.



### 3. Why a graphical model with latent variables can be a much harder problem?

In fully observed iid(independent and identically distributed random variables) settings, the log likelihood decomposes into a sum of local terms

$l_c(\theta;D)=\log p(x,z|\theta)=\log p(z|\theta_z)+\log p(x|z,\theta_x)$

However, with latent variables, all the parameters become coupled together via marginalization

$l_c(\theta;D)=\log\sum_zp(x,z|\theta)=\log\sum_zp(z|\theta_z)p(x|z,\theta_x)$

(So we need to exhaustion on every possible condition for latent variables)

Which is a much harder problem.



### 4. What is the key assumption for graphical model? Using HMM as an example, how much computational complexity has been reduced because of this assumption?

key assumption is that all variables are mutually independent.

1. condition only depends on its father (or ancestor for multi-layer HMM) nodes' condition
2. condition is independent to the time
3. output independence

DAG: node condition only depends on its father condition

For example, if we want to model n variables, each of which can have k values. If we set up a table, it can have the complexity of $O(k^n)$, however, if we use HMM to model it, suppose m represents the maximum variable numbers for a single conditional probability distribution, then the complexity of this graphical model is about $O(k^m)$. Since we can satisfy that $m<<n$, the computational complexity can be reduced greatly because of this assumption.



### 5. Why does EM not guarantee a global solution? What is a simple proof for that?

deep learning p386

Because it is possible for EM to converge to a local min, local max or a saddle point of the likelihood function(it is guaranteed to converge to a point with zero gradient).

Proof:

EM is composed of two steps: E step, which estimate some "missing" or "unobserved" data from observed data and current parameters; M step, which use this "complete" data to find the maximum likelihood parameter estimates(optimize a lower bound on the likelihood). It is obvious that the gap between the bounds and the likelihood function converges to zero, which is:

$\lim_{t\rightarrow\infin}L(\theta_t)=q_t(\theta_t)$

And the gradient of the bound also converges to the gradient of the likelihood function, which is:

$\lim_{t\rightarrow\infin}\nabla L(\theta_t)=\nabla b_t(\theta_t)$

since $\theta_t=\text{arg max}_\theta F(q^{t+1},\theta)$

we can have that:

$\nabla b_t(\theta_t)=0$, and therefore:

$\lim_{t-\rightarrow\infin}\nabla L(\theta_t)=0$

Hence it can only be guaranteed to converge to a point with zero gradient, which may not be a global solution.



### 6. Why is K-mean only an approximate and local solution for clustering?

K-means tries to minimize sum squared error(SSE), which is  not convex.

Because K-mean is an ill-posed method, which doesn't give us a single criteria to test its result in the real world, and we don't know if the result feature of K-mean clustering can represent data's feature in the real world. Also, since it doesn't consider the relationship between data, it can only converge to local solution and its result depends on the initial choice of clustering central point. Consequently, it is only an approximate and local solution for clustering.



### 7. How to interpret the HMM-based inference problem from a Bayesian perspective, using the forward/backward algorithm?

Suppose we now have the HMM model M and a sequence x, from which we want to infer the sequence y of states which maximizes $P(y|x,M)$ or the most probable subsequence of states. Here we should use forward/backward algorithm.

The forward algorithm can be used to calculate the forward probability:

$\alpha_t^k=P(x_1,...x_{t-1}, x_t, y_t^k=1)$

which can be calculated by:

$\alpha_t^k=P(x_t|y_t^k=1)\sum_i\alpha_{t-1}^i\alpha_{i,k}$

(then use dynamic programming to compute $\alpha_t^k$ for all k,t)

we want to calculate $P(y_t^k=1|x)$, which is:

$P(y_t^k=1,x)=P(x_1,..x_t, y_t^k=1)P(x_{t+1},...,x_T|y_t^k=1)$

In which the first term is forward $\alpha_t^k$ and the second term is backward $\beta_t^k$, which we can calculate by:

$\beta_t^k=\sum_i\alpha_{k,i}P(x_{t+1}|y_{t+1}^i=1)\beta_{t+1}^i$

Hence finally, we can infer the sequence y.



### 8. Show how to estimate a given hidden state for a given series of observations using the alpha and beta factors;

see 7.



### 9. For a Gaussian graphical model, what is the implication of sparsity for such a graphical model? How is such sparsity achieved computationally?

sparsity means most of events receive zero probability from the Gaussian graphical model, that is, many of the parameters in the graphical model is 0, so that the relation between features and events can be more clear.

To achieve such sparsity, we can apply Frobenius norm.

$||A||=F-\sqrt{\sum_{i=1}^m\sum_{j=1}^n|a_{ij}|^2}$





### 10. What would be the risk using a L1 as a relaxation for the sparsity estimation?

L1 is not differentiable every where(especially at the origin), needs to have some mathematical treatments.

L1 is easy to get the sparse result, which may lose some currently not-useful features that may be useful later.





## Dimension reduction and feature representation:

### 1. PCA is an example of dimensional reduction method; give a full derivation of PCA with respect to its eigenvectors; explain SVD and how it is used to solve PCA;

For a dataset that we want to apply PCA, we should first normalize them to zero sum and unit variance, than we calculate $\sum$ as:

$\sum=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}(x^{(i)})^T$

Then we need to find top k eigenvectors of $\sum$, which are $u_1, u_2,...u_k$. each eigenvalue corresponds to one eigenvector, and larger the eigenvalue is, more principle the eigenvector is. Then we can have:

$y^{(i)}=[u_1^Tx^{(i)}, u_2^Tx^{(i)},...,u_k^Tx^{(i)}]$

Then $y^{(i)}$ is the dimension-reduced data we can use.



SVD means single value decomposition, which can improve the complexity of computation for PCA from $O(N^2)$ to $O(N)$. It tells us that any matrix $X=R^{m\times n}$ can be decomposed into:

$X=U\sum V^T$

Where U is $m\times m$ diagonal, singular values of X, $U$ is $m\times m$, whose columns are Eigenvectors of $XX^T$ and V is $n\times n$, whose columns are eigenvectors of $X^TX$. In this way, to get the top k eigenvectors, we can just pick k top columns from $V$ for the $XX^T$.



### 2. Compare regular PCA with the low-ranked PCA, what would be advantage using the low-ranked PCA and how it is formulated?

After PCA, if we still use all of PCA result, it only find the orthogonal projections of the dataset, but not indeed give dimension reduction especially when n<p. Hence we need to select the top k components as our dimension-reduced data, which is called the low-ranked PCA. Its advantage is that we can improve the efficiency of our model calculation to a large extent while still keeps most of the useful information and get rid of the data that has no obvious relation to the labels.

Suppose we want to reduce dimension to p, we can choose only the first p eigenvectors based on the top p eigenvalues and then calculate the final dataset with only p dimensions.

 

### 3. For a low rank-regularized PCA, what would be the limit of dimension reduction for a given p and n of your data?

PCA can only find the linear relationships between data columns. So if the data is not linearly correlated, PCA doesn't perform well.  Also, PCA considers variances as somethings that can represent the importance of the data, however, it doesn't consider the distribution of data. What's more, for large p small n problems, where many of p may have no relations to the labels, PCA still gives result that are linear combinations of all input variables, which may lead to the wrong result in model building.



### 4. What is the key motivation (and contribution) behind deep learning, in terms of data representation?

Deep learning is based on learning multiple levels of representation, which corresponds to multiple levels of abstraction. We can treat deep learning as a kind of representation learning in which there are multiple levels of features, which can automatically discovered and composed together in the various levels to produce the output. Each level represents abstract features that are discovered from the features represented in the previous level, hence the level of abstraction increases with each level.

Deep learning tries first to decompose the problem, then recombine the problem, in order to learn the combination of these structures and concepts.



### 5. What would be the true features of an object modeling problem? Why does the feature decomposition in deep learning then a topological recombination could make a better sampling? What would be the potential problems making deep learning not a viable approach?

True features is the structures come from original features' decomposing and recombination.

In deep learning, true features are generated by model itself, which can transform the input that may not have close relation with the output into features that can have close relation with the output. Through multi-level calculation, the low-level features can be transformed into high-level features through the features decomposition and then topological recombination.

The potential problems is that:

1. the methods lack of theory, hence deep learning methods are often looked at as a black box, with most confirmations done empirically, rather than theoretically.
2. For classifying unrecognizable images, model generated by deep learning may misclassify them into a familiar category of ordinary images due to limitations in their internal representations.
3. Needs a lot of data to train
4. It is hard to explain the highly-abstract features generated by the model.

Another explanation of the potential problems:

1. there are a lot of non-linear relations combination inside the networks
2. the combination relations may not be restricted in one layer, however, they may have relation to any parameters in any layers.
3. If non-linear relations occurs, the deep learning model may only be able to converge to local optimization.



### 6. Explain the importance of appropriate feature selection being compatible with model selection in the context model complexity;

Different model and different learning methods may prefer different feature selections. For example, if we want to do the classification, we may prefer numeric features in the form of a feature vector, but when we are solving more complex tasks such as computer vision, a large more number of features, including highly-abstract features that generated by the model itself need to be included. Also, raw features can be redundant and too large to managed, hence we need to select principle information from these raw data to train our model.



### 7. What is the key motivation behind a kernel-based method in data representation?

Kernel-based method allows for arbitrary similarity measures to be used instead of only dot product, and allows people to learn over arbitrary instance spaces as long as the definition of a positive semi-definite kernel which can compute similarity between any two objects in the instance space. This means with kernel-based method, data can be represented not only in numerical way, but we can use this to dataset consists of strings, trees, graphs, etc.



(use kernels that depend on training data to fit the data distribution. If the distributions of the training data similar to the real-world data, then the kernel method fits well. Otherwise, over-fitting problem occurs)



### 8. What would be the ultimate and best representation for a high dimensional and complex problem?

There is no single certain best representation for all problem with high dimensional and complex problem. For each problem, we need to find the best way to extract and represent its features by applying  linear(PCA), nonlinear dimension reduction(Manifold Learning) or even features with higher abstract getting by deep learning algorithms. With the certain method, data for that problem can be best represented.



(A linear and convex representation would be ultimate, because linearity and convexity can reduce the complexity of the problem and guarantee some optimal solutions???)



### 9. Give two examples to highlight the importance of selecting appropriate dimensions for feature representations.

1. Use data to discriminate tiger and horse, dimension such as number of feet, heart and face organs may have no help.
2. Gene chip data.



### 10. For a typical big data problem (p>>n), what considerations we will have to take when trying to select an appropriate model(for instance, to perform a SVM)?

A lot of p may have no relations with the labels, so we should first select and extract the appropriate features based on professional knowledge(such as using kernel function), or do dimension reduction before the training such as PCA and manifold learning .

Also, regularization should be considered, in order to reduce the redundancy of the data and avoid high model complexity.

## General problems:

### 1. In learning, from the two key aspects, data and model, respectively, what are the key issues we normally consider in order to obtain a better model?



### 2. Why all machine learning problems are ill-posed?





### 3. Describe from the classification, to clustering, to HMM, to more complex graphical modeling, what we are trying to do for a more expressive model?





### 4. What are the potential risks we could take when trying to perform a logistic regression for classification using a sparsity-based regularization?



### 5. What are the potential risks we could take when trying to perform a linear regression using a sparsity-based regularization?



### 6. Give five different structural considerations a search can be constrained with corresponding simple scalars;



### 7. Give all universal, engineering, and computational principles that we have learned in this course to obtain both conceptually low-complexity model and computationally tractable algorithms?



### 8. Why data representation is at least equally as important as the actual modeling, the so-called representation learning?



### 9. How does the multiple-layer structure(deep learning) become attractive again?



### 10. What is the trend of AI research and development for the next 5-10 years?



## Other general problems

### 1. SVM is a linear classifier with a number of possible risks to be incurred, particularly with very high dimensional and overlapping problems. Use a simple and formal mathematics to show and justify:

(a) how a margin-based linear classifier like SVM can be even more robust than Logistic regression?

logistic regression is much more sensitive to the margin of different labels, and we may have to consider a distribution or model of the data before we start to train the model, which may bring more risk, hence SVM is more robust

(b)

first use dimension reduction, then use kernel function to control the overlapping boundary



### 2. Why a convolution-based deep learning might be a good alternative to address the dilemma of being more selective towards the features of an object, while remaining invariant toward anything else irrelevant to the aspect of interests?

(a) convolution-based deep learning doesn't use the apparent feature of an object selected by human. It can extract feature by itself from data, and learn about the object's feature related to the background environment, which is more important than feature itself, and won't be affected by this dilemma.

### Why a linear regression with regulations would result in features which are usually conceptually and structurally not meaningful?

(b) linear regression may use features that are not orthogonal to each other, are not normalized, and may even have much relationship with others. Unlike convolution-based deep learning, we can't make sure if feature we select for linear regression indeed have relation with their labels, hence results from these features may not been meaningful.



### 3. There are a number of nonlinear approaches to learn complex and high dimensional problems, including kernel and neural networks, please discuss the key differences in feature selection between these two alternatives, and their suitability

(a) kernel is one complex function applied to all data, while neural networks use a certain simple function to  separate those nonlinear data into many small parts.

### what are the major difficulties using a complex neural network as a non-linear classifier?

(b) 奇函数的选择

组合的方式

在考虑数据间关系的同时还要兼顾组合量的关系

既要使神经网络具有足够的表达能力，又要考虑数学计算上的能力（但会损失一定高阶的数学特征）



### 4. For any learning problems why a gradient-based search is much more favorable than other types of searches?

(a) 如果没有梯度的方向给定的确定性和唯一性， you have to 穷举各个方向的参数变化

(b) what would be the possible ramifications of having to impose some kinds of sequentiality in both providing data and observing results?



### 5. (a) use linear regression as the example to explain why L1 is more aggressive when trying to obtain sparser solutions compared to L2?

要求了解四个范数



### Under what conditions L1 might be a good approximation of the truth, which is L0?

(b) 

### 6. What is the key difference between a supervised vs. unsupervised learnings

(a) when an unlabeled data is presented, it has to been marginally estimated which label it should has. Although you can guess for many times, you can't make sure it is the global solution or it is a local solution.

### why unsupervised learning does not guaranty a global solution?

(b) search for mathematical formulas



### 7. For HMM, provide a Bayesian perspective about the forwarding message to enhance an inference

$\alpha$ recursion?

### how to design a more generalizable HMM which can still converge efficiently?

(b) the first several data should be the strongest features, which can give you the prior to deal with other data.



### 8. Using a more general graphical model to discuss the depth of a developing prior-distribution as to its contribution for a possible inference; 

(a)

depth increase, 



### how local likelihoods can be used as the inductions to facilitate the developing inference?

(b)

结果和先验相符



### 9. Learning from observation is an ill-posed problem, however we still work on it and even try to obtain convex, linear, and possibly generalizable solutions. Please discuss what key strategies in data mining we have developed that might have remedied the ill-posed nature at least in part?

(a)

有选择地选择feature, 但又在不是穷举的情况下以保证模型有一定的泛化能力



### Why in general linear model are more robust than more complex ones?

(b)



### 10. Using logistic regression and likelihood estimation for learning a mixture model(such as the Gaussian Mixture Model), please using Bayesian perspective to discuss the differences and consistencies of the two approaches;



数学形式+解释



### Why logistic function is a universal posterior for many mixture models?