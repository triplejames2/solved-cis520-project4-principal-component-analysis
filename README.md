Download Link: https://assignmentchef.com/product/solved-cis520-project4-principal-component-analysis
<br>
<ol>

 <li><strong>Principal Component Analysis. </strong>In this exercise, you are provided part of the MNIST digit data set, containing 3,600 images of handwritten digits (data is provided in P1/X_train.csv). Each example is a 28 × 28 grayscale image, leading to 784 pixels. These images correspond to 3 different digits (’3’, ’6’, and ’9’), although you do not have the labels for them. You will perform PCA in an unsupervised manner.

  <ul>

   <li>To get familiar with what the data looks like, generate the last example (the 3<em>,</em>600<em><sup>th </sup></em>row of X_train) as an image. Provide your code and the resulting image. (<em>Hint: </em>You can reshape the data back into a 28 × 28 image using reshape in NumPy, and you can use imshow with cmap=’gray’ to see the picture. Make sure you are able to get the picture to face the right way, which should look like a ”9”.)</li>

   <li>Implement PCA using SVD and run it on the data. You can use the function linalg.svd in NumPy. (<em>Reminder: </em>Make sure you understand the NumPy function you used, especially whether the function standardizes the data set before running SVD. Make sure that the data is mean-centered.) Just like in the previous part, generate the first principal component vector as an image. Repeat for the second and third. Do these images make sense? Explain.</li>

   <li>Create a 2D scatter plot showing all the data points projected in the first 2 PC dimensions, clearly labeling your axes. Make sure the data points are mean-centered before projecting onto the principal components. Now create a similar plot showing the same data points projected in the 100<em><sup>th </sup></em>and 101<em><sup>st </sup></em>PC dimensions. What do you observe? Can you explain why you might expect this?</li>

   <li>Graph the (in-sample) fractional reconstruction accuracy as a function of the number of principal components that are included. Also give a table listing the number of principal components needed to achieve each of 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, and 10% reconstruction accuracy (i.e., to explain X% of the variance).</li>

   <li>Using the numbers you found in the previous part, reconstruct the 1000<em><sup>th</sup></em>, 2000<em><sup>th</sup></em>, and 3000<em><sup>th </sup></em>examples using each of the different numbers of principal components. (<em>Hint: </em>We have provided a function plot_reconstruction for your convenience. Make sure you read the documentation within this function to understand its input arguments.) For instance, the last example looks like:<em>                                                                                                 </em>3</li>

  </ul></li>

</ol>

<em>Python Instructions:</em>

<ul>

 <li><em>Please follow the instructions in the comments on input/output specifications. Be aware that the dimension of your vector variables can be a source of bugs.</em></li>

 <li><em>Include all of the codes you used for experiments and plotting in </em>PS4-P1.py <em>and follow submission instructions when submitting.</em></li>

</ul>

<ol start="2">

 <li><strong>EM Practice: Red and Blue Coins. </strong>Your friend has two coins: a red coin and a blue coin, with biases <em>p<sub>r </sub></em>and <em>p<sub>b</sub></em>, respectively (i.e. the red coin comes up heads with probability <em>p<sub>r</sub></em>, and the blue coin does so with probability <em>p<sub>b</sub></em>). She also has an inherent preference <em>π </em>for the red coin. She conducts a sequence of <em>m </em>coin tosses: for each toss, she first picks either the red coin with probability <em>π </em>or the blue coin with probability 1−<em>π</em>, and then tosses the corresponding coin; the process for each toss is carried out independently of all other tosses. You don’t know which coin was used on each toss; all you are told are the outcomes of the <em>m </em>tosses (heads or tails). In particular, for each toss <em>i</em>, define a random variable <em>X<sub>i </sub></em>as</li>

</ol>

(

1         if the <em>i</em>-th toss results in heads

<em>X<sub>i </sub></em>=

<ul>

 <li></li>

</ul>

Then the data you see are the values <em>x</em><sub>1</sub><em>,…,x<sub>m </sub></em>taken by these <em>m </em>random variables. Based on this data, you want to estimate the parameters <em>θ </em>= (<em>π,p<sub>r</sub>,p<sub>b</sub></em>). To help with this, for each toss <em>i</em>, define a latent (unobserved) random variable <em>Z<sub>i </sub></em>as follows:

(

<ul>

 <li>if the <em>i</em>-th toss used the red coin <em>Z<sub>i </sub></em>=</li>

</ul>

0       otherwise.

<ul>

 <li>Let <em>X </em>be a random variable denoting the outcome of a coin toss according to the process described above, and let <em>Z </em>be the corresponding latent random variable indicating which coin was used, also as described above (both <em>X </em>and <em>Z </em>take values in {0<em>,</em>1} as above). Write an expression for the joint distribution of <em>X </em>and <em>Z</em>. Give your answer in the form</li>

</ul>

<em>p</em>(<em>x,z</em>; <em>θ</em>) =   <em>.</em>

<ul>

 <li>Write an expression for the complete-data log-likelihood, ln</li>

 <li>Suppose you knew the values <em>z<sub>i </sub></em>taken by the latent variables <em>Z<sub>i</sub></em>. What would be the maximumlikelihood parameter estimates <em>θ</em><sub>b</sub>? Give expressions for <em>π</em><sub>b</sub>, <em>p</em><sub>b</sub><em>r</em>, and <em>p</em><sub>b</sub><em>b </em>(in terms of <em>x<sub>i </sub></em>and <em>z<sub>i</sub></em>). Show your calculations.</li>

 <li>In the absence of knowledge of <em>z<sub>i</sub></em>, one possibility for estimating <em>θ </em>is to use the EM algorithm. Recall that the algorithm starts with some initial parameter estimates <em>θ</em><sup>0</sup>, and then on each iteration <em>t</em>, performs an E-step followed by an M-step. Let <em>θ<sup>t </sup></em>denote the parameter estimates at the start of iteration <em>t</em>. In the E-step, for each toss <em>i</em>, the algorithm requires computing the posterior distribution of the latent variable <em>Z<sub>i </sub></em>under the current parameters <em>θ<sup>t</sup></em>. Calculate the posterior probability <strong>P</strong>(<em>Z<sub>i </sub></em>= 1|<em>X<sub>i </sub></em>= <em>x<sub>i</sub></em>; <em>θ<sup>t</sup></em>).</li>

 <li>For each toss <em>i</em>, denote the posterior probability computed in part (d) above by <em>γ<sub>i</sub><sup>t </sup></em>(so that <em>γ<sub>i</sub><sup>t </sup></em>= <strong>P</strong>(<em>Z<sub>i </sub></em>= 1|<em>X<sub>i </sub></em>= <em>x<sub>i</sub></em>; <em>θ<sup>t</sup></em>)). Then the expected complete-data log-likelihood with respect to these posterior distributions is</li>

</ul>

<em> .</em>

The M-step of the EM algorithm requires finding parameters <em>θ<sup>t</sup></em><sup>+1 </sup>that maximize this expected complete-data log-likelihood. Determine the updated parameters <em>θ<sup>t</sup></em><sup>+1</sup>. Give expressions for <em>π<sup>t</sup></em><sup>+1</sup>, , and (in terms of <em>x<sub>i </sub></em>and <em>γ<sub>i</sub><sup>t</sup></em>). Show your calculations.

<ol start="3">

 <li><strong>Programming Exercise: Gaussian Mixture Models. </strong>Write a piece of Python code to implement the EM algorithm for learning a Gaussian mixture model (GMM) given a data set <strong>X</strong>, where <strong>X </strong>is an <em>m</em>×<em>d </em>matrix (<em>m </em>instances, each of dimension <em>d</em>), and a target number of Gaussians <em>K</em>: your program should take the training data <strong>X </strong>and the target number of Gaussians <em>K </em>as input and should output estimated parameters of a mixture of <em>K </em>Gaussians (specifically, it should output mixing coefficients <em>π</em>b ∈ ∆<em><sub>K</sub></em>, mean vectors <em><sup>µ</sup></em><sub>b</sub>1<em>,…,</em><em><sup>µ</sup></em><sub>b</sub><em>K </em><sup>∈ </sup>R<em><sup>d</sup></em>, and covariance matrices <strong>Σ</strong><sub>b1</sub><em>,…,</em><strong>Σ</strong><sub>b<em>K </em></sub><sup>∈ </sup>R<em><sup>d</sup></em><sup>×<em>d</em></sup>). For this problem, you are provided a 2-dimensional data set generated from a mixture of (3) Gaussians. The data set is divided into training and test sets; the training set has 480 data points and the test set has 120 data points. The goal is to learn a GMM from the training data; the quality of the learned GMM will be measured via its log-likelihood on the test data.</li>

</ol>

<strong>EM initialization and stopping criterion: </strong>Initialize the mixing coefficients to a uniform distribution, and each of the covariance matrices to be the <em>d </em>× <em>d </em>identity matrix: <em>π</em><sup>0 </sup>= (1<em>/K,…,</em>1<em>/K</em>)<sup>&gt;</sup>, and

. Initializations for the means will be provided to you (if you like, you can write

your program to take the mean initialization as an additional input). For the stopping criterion, on each iteration <em>t</em>, keep track of the (incomplete-data) log-likelihood on the training data, ); stop when either the change in log-likelihood,), becomes smaller than 10<sup>−6</sup>, or when the number of iterations reaches 1000 (whichever occurs earlier; here <em>θ<sup>t </sup></em>denotes the parameter estimates on iteration <em>t</em>).

<ul>

 <li><strong>Known </strong><em>K</em><strong>. </strong>For this part, assume you are given <em>K </em>= 3.

  <ol>

   <li><strong>Learning curve. </strong>Use your implementation of EM for GMMs to learn a mixture of 3 Gaussians from increasing fractions of the training data (10% of the training data, then 20% of the training data, then 30% and so on upto 100%). You must use the subsets provided in the folder P3/TrainSubsets; in each case, to initialize the means of the Gaussians, use the initializations provided in the folder P3/MeanInitialization. In each case, calculate the <strong>normalized </strong>(incomplete-data) log-likelihood of the learned model on the training data, as well as the normalized log-likelihood on the test data (here normalizing means dividing the log-likelihood by the number of data points). In a single plot, give curves showing the normalized train and test log-likelihoods (on the <em>y</em>-axis) as a function of the fraction of data used for training (on the <em>x</em>-axis). (Note: the training log-likelihood should be calculated only on the subset of examples used for training, not on all the training examples available in the given data set.)</li>

   <li><strong>Analysis of learned models. </strong>For each of the learned models (corresponding to the different subsets of the training data), show a plot of the Gaussians in the learned GMM overlaid on the test data. What do you observe? For the final model (learned from 100% of the training data), also write down all the parameters of the learned GMM, together with the (normalized) train and test log-likelihoods.</li>

  </ol></li>

</ul>

<em>Hints:</em>

<ol>

 <li><em>You may use the function we provide in </em>plot multiple contour plots <em>to assist in plotting the density curves. Be sure to carefully read the function header, so you know how to structure your inputs.</em></li>

</ol>

<ul>

 <li><strong>Unknown </strong><em>K</em><strong>. </strong>Now suppose you do not know the right number of Gaussians in the mixture model to be learned. Select the number of Gaussians <em>K </em>from the range {1<em>,…,</em>5} using 5-fold crossvalidation on the full training data (use the folds provided in the folder P3/CrossValidation). For each <em>K</em>, to initialize the means of the <em>K </em>Gaussians, use the initializations provided in the folder P3/MeanInitialization. In a single plot, draw 3 curves showing the (normalized) train, test, and cross-validation log-likelihoods (on the <em>y</em>-axis) as a function of number of Gaussians <em>K </em>(on the <em>x</em>-axis); Write down the chosen value of <em>K </em>(with the highest cross-validation log-likelihood).</li>

</ul>

<em>Python Instructions:</em>

<em>                                                                                      </em>

<ul>

 <li><em>Please follow the instructions in the comments on input/output specifications. Be aware that the dimension of your vector variables can be a source of bugs.</em></li>

 <li><em>Implement the </em>E step<em>, </em>M step<em>, and </em>compute llh <em>methods in </em>py<em>. You may use the function </em>scipy.stats.multivariate normal <em>in your implementation.</em></li>

 <li><em>Include all of the codes you used for experiments and plotting in </em>PS4-P3.py <em>and follow submission instructions when submitting.</em></li>

</ul>

<ol start="4">

 <li>(35 points) <strong>Hidden Markov Models. </strong>On any given day, Alice is in one of two states: happy or sad. You do not know her internal state, but get to observe her activities in the evening. Each evening, she either sings, goes for a walk, or watches TV.</li>

</ol>

Alice’s state on any day is random. Her state <em>Z</em><sub>1 </sub>on day 1 is equally likely to be happy or sad:

<em>P</em>(<em>Z</em><sub>1 </sub>= happy) = 1<em>/</em>2<em>.</em>

Given her state <em>Z<sub>t </sub></em>on day <em>t</em>, her state <em>Z<sub>t</sub></em><sub>+1 </sub>on the next day is governed by the following probabilities (and is conditionally independent of her previous states and activities):

<em>P</em>(<em>Z<sub>t</sub></em><sub>+1 </sub>= happy |<em>Z<sub>t </sub></em>= happy) = 4<em>/</em>5                        <em>P</em>(<em>Z<sub>t</sub></em><sub>+1 </sub>= happy |<em>Z<sub>t </sub></em>= sad) = 1<em>/</em>2<em>.</em>

Alice’s activities are also random. Her activities vary based on her state; given her state <em>Z<sub>t </sub></em>on day <em>t</em>, her activity <em>X<sub>t </sub></em>on that day is governed by the following probabilities (and is conditionally independent of everything else):

<em>P</em>(<em>X<sub>t </sub></em>= sing |<em>Z<sub>t </sub></em>= happy) = 5<em>/</em>10                       <em>P</em>(<em>X<sub>t </sub></em>= sing |<em>Z<sub>t </sub></em>= sad) = 1<em>/</em>10

<em>P</em>(<em>X<sub>t </sub></em>= walk|<em>Z<sub>t </sub></em>= happy) = 3<em>/</em>10      <em>P</em>(<em>X<sub>t </sub></em>= walk|<em>Z<sub>t </sub></em>= sad) = 2<em>/</em>10 <em>P</em>(<em>X<sub>t </sub></em>= TV |<em>Z<sub>t </sub></em>= happy) = 2<em>/</em>10             <em>P</em>(<em>X<sub>t </sub></em>= TV |<em>Z<sub>t </sub></em>= sad) = 7<em>/</em>10<em>.</em>

<ul>

 <li>(15 points) Suppose you observe Alice singing on day 1 and watching TV on day 2, i.e. you observe <em>x</em><sub>1:2 </sub>= (sing<em>,</em>TV). Find the joint probability of this observation sequence together with each possible hidden state sequence that could be associated with it, i.e. find the four probabilities below. Show your calculations.</li>

</ul>

<em>P X</em><sub>1:2 </sub>= (sing<em>,</em>TV)<em>,Z</em><sub>1:2 </sub>= (happy<em>,</em>happy) <em>P X</em><sub>1:2 </sub>= (sing<em>,</em>TV)<em>,Z</em><sub>1:2 </sub>= (happy<em>,</em>

) <em>P X</em><sub>1:2 </sub>= (sing<em>,</em>TV)<em>,Z</em><sub>1:2 </sub>= (sad<em>,</em>sad)

Based on these probabilities, what is the most likely hidden state sequence <em>z</em><sub>1:2</sub>? What is the individually most likely hidden state on day 2 ?

<ul>

 <li> Write a small piece of Python code to implement the Viterbi algorithm, and use this to calculate both the most likely hidden state sequence <em>z</em><sub>1:10 </sub>and the corresponding maximal joint probability <em>p</em>(<em>x</em><sub>1:10</sub><em>,z</em><sub>1:10</sub>) for each of the following observation sequences:</li>

</ul>

<em>x</em><sub>1:10 </sub>= (sing<em>,</em>walk<em>,</em>TV<em>,</em>sing<em>,</em>walk<em>,</em>TV<em>,</em>sing<em>,</em>walk<em>,</em>TV<em>,</em>sing) <em>x</em><sub>1:10 </sub>= (sing<em>,</em>sing<em>,</em>sing<em>,</em>TV<em>,</em>TV<em>,</em>TV<em>,</em>TV<em>,</em>TV<em>,</em>TV<em>,</em>TV) <em>x</em><sub>1:10 </sub>= (TV<em>,</em>walk<em>,</em>TV<em>,</em>sing<em>,</em>sing<em>,</em>sing<em>,</em>walk<em>,</em>TV<em>,</em>sing<em>,</em>sing)

In your code, rather than multiplying probabilities (which can lead to underflow), you may find it helpful to add their logarithms (and later exponentiate to obtain the final result). Include a snippet of your code in your L<sup>A</sup>TEX submission.

<em>Python Instructions:</em>

<ul>

 <li><em>Please follow the instructions in the comments on input/output specifications. Be aware that the dimension of your vector variables can be a source of bugs.</em></li>

 <li><em>Include your implementation of the Viterbi algorithm </em>viterbi <em>and all of the codes you used for experiments and plotting in </em>PS4-P4.py <em>and follow submission instructions when submitting.</em></li>

</ul>

<ol start="5">

 <li><strong>Bayesian Networks. </strong>Consider the Bayesian network over 8 random variables <em>X</em><sub>1</sub>, <em>X</em><sub>2</sub>, <em>X</em><sub>3</sub>, <em>X</em><sub>4</sub>, <em>X</em><sub>5</sub>, <em>X</em><sub>6</sub>, <em>X</em><sub>7</sub>, <em>X</em><sub>8 </sub>shown below (assume for simplicity that each random variable takes 2 possible values):</li>

</ol>

<ul>

 <li>Write an expression for the joint probability mass function <em>p</em>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,x</em><sub>3</sub><em>,x</em><sub>4</sub><em>,x</em><sub>5</sub><em>,x</em><sub>6</sub><em>,x</em><sub>7</sub><em>,x</em><sub>8</sub>) that makes the same (conditional) independence assumptions as the Bayesian network above.</li>

 <li>Consider a joint probability distribution satisfying the following factorization:</li>

</ul>

<em>p</em>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,x</em><sub>3</sub><em>,x</em><sub>4</sub><em>,x</em><sub>5</sub><em>,x</em><sub>6</sub><em>,x</em><sub>7</sub><em>,x</em><sub>8</sub>) = <em>p</em>(<em>x</em><sub>1</sub>)<em>p</em>(<em>x</em><sub>2</sub>)<em>p</em>(<em>x</em><sub>3</sub>)<em>p</em>(<em>x</em><sub>4 </sub>|<em>x</em><sub>1</sub>)<em>p</em>(<em>x</em><sub>5 </sub>|<em>x</em><sub>2</sub><em>,x</em><sub>3</sub>)<em>p</em>(<em>x</em><sub>6 </sub>|<em>x</em><sub>3</sub><em>,x</em><sub>5</sub>)<em>p</em>(<em>x</em><sub>7</sub>)<em>p</em>(<em>x</em><sub>8 </sub>|<em>x</em><sub>5</sub>)<em>.</em>

Is this distribution included in the class of joint probability distributions that can be represented by the Bayesian network above? Briefly explain your answer.

<ul>

 <li>If the edge from <em>X</em><sub>5 </sub>to <em>X</em><sub>8 </sub>is removed from the above network, will the class of joint probability distributions that can be represented by the resulting Bayesian network be smaller or larger than that associated with the original network? Briefly explain your answer.</li>

 <li>Given the above figure, determine whether each of the following is true or false. Briefly justify your answer.</li>

</ul>

<ol>

 <li><em>X</em><sub>4 </sub>⊥ <em>X</em><sub>5 </sub>ii. <em>X</em><sub>2 </sub>⊥ <em>X</em><sub>6</sub></li>

</ol>

iii. <em>X</em><sub>3 </sub>⊥ <em>X</em><sub>4 </sub>|<em>X</em><sub>8 </sub>iv. <em>X</em><sub>2 </sub>⊥ <em>X</em><sub>8 </sub>|<em>X</em><sub>5</sub>