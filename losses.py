import numpy as np 
import tensorflow as tf


def L1_Loss(y_true,y_pred):
    return np.sum(np.abs(y_true-y_pred),axis=-1)
def L2_Loss(y_true,y_pred):
    return np.sum((y_true-y_pred)**2,axis=-1)
def MSE(y_true,y_pred):
    """
    Return Mean Squared Error loss function given y_true , y_pred .

    Args:
      
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Float): overall scalar loss summed across all classes
    """
    y_true=np.cast[str(y_pred.dtype).split("'")[1]](y_true)
    return np.mean((np.square(y_true-y_pred)),axis=-1)
def MAE(y_true,y_pred):
    y_true=np.cast[str(y_pred.dtype).split("'")[1]](y_true)
    return np.mean((np.absolute(y_true-y_pred)),axis=-1)

def KLD(y_true,y_pred):
    y_true=np.cast[str(y_pred.dtype).split("'")[1]](y_true)
    y_true=np.clip(y_true,1e-07,1)
    y_pred=np.clip(y_pred,1e-07,1)
    return np.sum(y_true*np.log(y_true/y_pred),axis=-1)



def logcosh(y_true,y_pred):
    def softplus(x):
        return  np.log(np.ones_like(x) + np.exp(x))

    def _logcosh(x):
        return x +  softplus(-2. *x) - np.cast[str(x.dtype).split("'")[1]](np.log(2.))
    return  np.mean(_logcosh(y_pred - y_true), axis=-1)


#$%^
def hinge(y_true,y_pred):

    loss=np.mean(np.maximum(1.0-y_true * y_pred,0.0),axis=-1)
    return loss

def huber(y_true,y_pred,delta=1.0,reduction=True):
    """
    Args:
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
      reduction: True to apply to loss. Default value is True. True indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`.
        if reduction is False is will overall scalar loss summed across all classes

        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Float): overall scalar loss summed across all classes
    """
  
    if reduction ==False:

        error = np.subtract(y_pred, y_true)
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = np.subtract(abs_error, quadratic)
        return np.mean(
            np.add(
             np.multiply(
                 0.5,
                  np.multiply(quadratic, quadratic)),
                np.multiply(delta, linear)),
                axis=-1)
    if reduction ==True:
        error = np.subtract(y_pred, y_true)
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = np.subtract(abs_error, quadratic)
        return np.mean(
            np.add(
             np.multiply(
                 0.5,
                  np.multiply(quadratic, quadratic)),
                np.multiply(delta, linear)))
def MAPE(y_true,y_pred):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.
  `loss = 100 * mean(abs(y_true - y_pred) / y_true, axis=-1)`


  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
  Returns:
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
  """

    diff = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-7))
    return 100. * np.mean(diff, axis=-1)


def MSLE(y_true,y_pred):

    first_log = np.log(np.maximum(y_pred,1e-7) + 1.)
    second_log = np.log(np.maximum(y_true,1e-7) + 1.)
    return np.mean(np.square(first_log- second_log), axis=-1)




def poisson(y_true,y_pred):
    """Computes the Poisson loss between y_true and y_pred.
  The Poisson loss is the mean of the elements of the `Tensor`
  `y_pred - y_true * log(y_pred)`.


    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
        Poisson loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    return np.mean(y_pred-y_true*np.log(y_pred+1e-7),axis=-1)



#$%^
def squared_hinge(y_true,y_pred):

    return np.mean(np.square(np.maximum(1.0-y_true*y_pred,0.0)),axis=-1)

#$%^  
def quantile(y_true,y_pred,Y=0.25):
    q=Y
    e = y_pred-y_true
    loss=np.mean(np.maximum(q*e, (q-1)*e),axis=-1)
    #loss=np.mean(np.dot((0.25-1),np.abs(y_true-y_pred)),axis=-1)+np.mean(np.dot(0.25,np.abs(y_true-y_pred)),axis=-1)
    return loss

def binary_crossentropy(y_true,y_pred,epsilon=1e-7):
    """
    Return binary_crossentropy loss function given y_true , y_pred .

    Args:
      
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Float): overall scalar loss summed across all classes
    """
    y_true=np.cast[str(y_pred.dtype).split("'")[1]](y_true)
    return -np.mean((y_true*np.log(y_pred+epsilon)+(1-y_true)*np.log(1-y_pred+epsilon)),axis=-1)



#$%^
def binary_crossentropy_weighted_loss(y_true, y_pred,labels, epsilon=1e-7):
    """
    Return weighted loss function given y_true , y_pred , labels.

    Args:
      
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        labels (Tensor): Tensor of the labels 
    Returns:
        loss (Float): overall scalar loss summed across all classes
    """
    y_true=np.cast[str(y_pred.dtype).split("'")[1]](y_true)
    N = labels.shape[0]
    
    pos_weights = np.sum(labels==1,axis=0)/N
    neg_weights = np.sum(labels==0,axis=0)/N
    
    
    # initialize loss to zero
    loss = 0.0
        

    # for each class, add average weighted loss for that class 
    for i in range(len(pos_weights)):
        loss +=-np.mean((pos_weights*y_true*np.log(y_pred+epsilon))+(neg_weights*(1-y_true)*np.log(1-y_pred+epsilon)),axis=-1)
        
    
    return loss 
def soft_dice_loss(y_true, y_pred, axis=(1,2,3), 
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the np.sum
                            and np.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """


    dice_numerator = 2*np.sum(y_true*y_pred,axis=axis)+epsilon
    dice_denominator = np.sum(y_true**2,axis=axis)+np.sum(y_pred**2,axis=axis)+epsilon
    dice_loss = 1-np.mean((dice_numerator/dice_denominator))


    return dice_loss




#test 
a=tf.convert_to_tensor([[1,0,0,1],[1,0,1,0]],dtype=tf.float32)
b=tf.convert_to_tensor([[0.91,0.01,0.01,0.9],
                            [0.7,0.05,0.8,0.01]],dtype=tf.float32)


print(tf.nn.l2_loss(a,b))
print(L1_Loss(a,b))
