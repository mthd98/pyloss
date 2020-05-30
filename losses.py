import tensorflow as tf
import numpy as np 


def MSE(y_true,y_pred):
    """
    Return Mean Squared Error loss function given y_true , y_pred .

    Args:
      
        y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
        y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
    Returns:
        loss (Float): overall scalar loss summed across all classes
    """
    y_true=tf.cast(y_true,y_pred.dtype)
    return tf.math.reduce_mean((tf.math.square(y_true-y_pred)),axis=-1)
def MAE(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)
    return tf.reduce_mean((tf.math.abs(y_true-y_pred)),axis=-1)

def KLD(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)
    
    y_true=tf.clip_by_value(y_true,1e-07,1)
    y_pred=tf.clip_by_value(y_pred,1e-07,1)
    return tf.reduce_sum(y_true*tf.math.log(y_true/y_pred),axis=-1)



def logcosh(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)
    def _logcosh(x):
        return x +  tf.nn.softplus(-2. *x) - tf.cast(tf.math.log(2.),x.dtype)
    return  tf.reduce_mean(_logcosh(y_pred - y_true), axis=-1)


#$%^
def hinge(y_true,y_pred):
    
    y_true=tf.cast(y_true,y_pred.dtype)
    loss=tf.reduce_mean(tf.maximum(1.0-y_true * y_pred,0.0),axis=-1)
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
    y_true=tf.cast(y_true,y_pred.dtype)
    if reduction ==False:

        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = tf.subtract(abs_error, quadratic)
        return tf.reduce_mean(
            tf.add(
             tf.multiply(
                 0.5,
                  tf.multiply(quadratic, quadratic)),
                tf.multiply(delta, linear)),
                axis=-1)
    if reduction ==True:
        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = tf.subtract(abs_error, quadratic)
        return tf.reduce_mean(
            tf.add(
             tf.multiply(
                 0.5,
                  tf.multiply(quadratic, quadratic)),
                tf.multiply(delta, linear)))
def MAPE(y_true,y_pred):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.
  `loss = 100 * mean(abs(y_true - y_pred) / y_true, axis=-1)`


  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
  Returns:
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
  """
    y_true=tf.cast(y_true,y_pred.dtype)
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-7))
    return 100. * tf.reduce_mean(diff, axis=-1)


def MSLE(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)
    first_log = tf.math.log(tf.maximum(y_pred,1e-7) + 1.)
    second_log = tf.math.log(tf.maximum(y_true,1e-7) + 1.)
    return tf.reduce_mean(tf.square(first_log- second_log), axis=-1)




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
    y_true=tf.cast(y_true,y_pred.dtype)
    return tf.reduce_mean(y_pred-y_true*tf.math.log(y_pred+1e-7),axis=-1)



#$%^
def squared_hinge(y_true,y_pred):
    y_true=tf.cast(y_true,y_pred.dtype)

    return tf.reduce_mean(tf.square(tf.maximum(1.0-y_true*y_pred,0.0)),axis=-1)

#$%^  
def quantile(y_true,y_pred,Y=0.25):
    y_true=tf.cast(y_true,y_pred.dtype)
    q=Y
    e = y_pred-y_true
    loss=tf.reduce_mean(tf.maximum(q*e, (q-1)*e),axis=-1)
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
    y_true=tf.cast(y_true,y_pred.dtype)
    return -tf.reduce_mean((y_true*tf.math.log(y_pred+epsilon)+(1-y_true)*tf.math.log(1-y_pred+epsilon)),axis=-1)



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
    y_true=tf.cast(y_true,y_pred.dtype)
    N = labels.shape[0]
    
    pos_weights = np.sum(labels==1,axis=0)/N
    neg_weights = np.sum(labels==0,axis=0)/N
    
    
    # initialize loss to zero
    loss = 0.0
        

    # for each class, add average weighted loss for that class 
    for i in range(len(pos_weights)):
        loss +=-tf.reduce_mean((pos_weights*y_true*tf.math.log(y_pred+epsilon))+(neg_weights*(1-y_true)*tf.math.log(1-y_pred+epsilon)),axis=-1)
        
    
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
                      Hint: pass this as the 'axis' argument to the tf.reduce_sum
                            and tf.reduce_mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """

    y_true=tf.cast(y_true,y_pred.dtype)
    dice_numerator = 2*tf.reduce_sum(y_true*y_pred,axis=axis)+epsilon
    dice_denominator = tf.reduce_sum(y_true**2,axis=axis)+tf.reduce_sum(y_pred**2,axis=axis)+epsilon
    dice_loss = 1-tf.reduce_mean((dice_numerator/dice_denominator))


    return dice_loss



def categorical_crossentropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_true=tf.cast(y_true,y_pred.dtype)
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred =  tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    # calc
    loss = y_true * tf.math.log(y_pred) 
    loss = -tf.reduce_sum(loss, -1)
    return tf.abs(loss)
#$%^
def yolo_v2_loss(y_true, y_pred,GRID_W,GRID_H,BATCH_SIZE,ANCHORS,BOX,COORD_SCALE,NO_OBJECT_SCALE,OBJECT_SCALE,CLASS_WEIGHTS,CLASS_SCALE,WARM_UP_BATCHES):
    
    mask_shape = tf.shape(y_true)[:4]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    
    true_box_conf = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)
    
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """    
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    
    return loss
