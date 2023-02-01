import tensorflow as tf
from  tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K

import  numpy as np

class WeightedAverage(layers.Layer):
    def __init__(self,dims=2,name=None):
        super(WeightedAverage, self).__init__(name=name)
        # tf.compat.v1.set_random_seed(seed)
        # tf.keras.utils.set_random_seed(seed)
        # tf.random.set_seed(seed)
        self.dims=dims
    
    def build(self, input_shape):
        self.W = self.add_weight("input_weight",shape=(self.dims,),dtype=tf.float32,trainable=True,initializer=keras.initializers.ones())
        
    def call(self,inputs):
        res=inputs[0]*self.W[0]
        ws=self.W[0]
        for i in range(1,self.dims):
            res+=inputs[i]*self.W[i]
            ws+=self.W[i]
        
        return res/ws

class GroupScore(layers.Layer):
    def __init__(self,nclusters,name=None):#seed):
        self.nclusters=nclusters
        super(GroupScore, self).__init__(name=name)
        # tf.compat.v1.set_random_seed(seed)
        # tf.keras.utils.set_random_seed(seed)
        # tf.random.set_seed(seed)
        
    def call(self,inputs):
        nclusters=self.nclusters
        pr = inputs[0]
        vr=tf.stop_gradient(inputs[1])

        vr=tf.squeeze(vr,-1)

        vmeans,vmaxs,vmins=[],[],[]
        for g in range(nclusters):
            tmp = tf.reshape(tf.where(tf.equal(vr, g), pr, np.nan), [-1])
            tmp = tf.gather(tmp, tf.where(tf.logical_not(tf.math.is_nan(tmp))))

            vmeans.append(K.mean(tmp))
            vmins.append(K.min(tmp))
            vmaxs.append(K.max(tmp))

        vmeans=tf.stack(vmeans)
        vmins=tf.stack(vmins)
        vmaxs=tf.stack(vmaxs)

        #To avoid nans for groups with single item
        no_scale_idx=tf.equal(vmins,vmaxs)
        vmins=tf.gather(tf.where(no_scale_idx,tf.cast(0.0,dtype=vmins.dtype),vmins),vr)
        vmaxs=tf.gather(tf.where(no_scale_idx,tf.cast(1.0,dtype=vmaxs.dtype),vmaxs),vr)


        #Replace nan with min/2 for Convulation
        nan_idx = tf.math.is_nan(vmeans)
        vmeans = tf.where(nan_idx, K.min(tf.gather(vmeans, tf.where(tf.logical_not(nan_idx)))) / 2, vmeans)

        sort_ids = tf.argsort(vmeans)
        orig_ids = tf.argsort(sort_ids)

        vmeans_sorted = tf.gather(vmeans, sort_ids)

        conv_data = tf.concat([vmeans_sorted[:1] , vmeans_sorted, vmeans_sorted[-1:] * 2], axis=0)
        conv_k = tf.ones(2, dtype=conv_data.dtype)

        #mm lower limit
        f0 = tf.squeeze(
            tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 1.99, [-1, 1, 1]), 1, padding="VALID")[
                0][:-1])

        #mm upper limit
        f1 = tf.squeeze(
            tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 2.01, [-1, 1, 1]), 1, padding="VALID")[
                0][1:])

        f0=tf.gather(tf.where(no_scale_idx,tf.cast(0.0,dtype=f0.dtype),tf.gather(f0, orig_ids)),vr)
        f1=tf.gather(tf.where(no_scale_idx,tf.cast(1.0,dtype=f1.dtype),tf.gather(f1, orig_ids)),vr)
        
        ##Min Max Scaling
        tmp=((tf.stop_gradient(pr)-tf.stop_gradient(vmins))/tf.stop_gradient(vmaxs - vmins)) *  tf.stop_gradient(f1 - f0) + tf.stop_gradient(f0)
        
        scale = pr/tmp
        scale = tf.where(tf.logical_or(tf.math.is_nan(scale),tf.math.is_inf(scale)), 0.0, scale)

        res = pr * tf.stop_gradient(scale)
        
        return tf.ensure_shape(res,[None,pr.shape[1]])

# def lambda_group_score(inputs,nclusters):
#         # nclusters=self.nclusters
#         pr = inputs[0]
#         vr=tf.stop_gradient(inputs[1])

#         vr=tf.squeeze(vr,-1)

#         vmeans,vmaxs,vmins=[],[],[]
#         for g in range(nclusters):
#             tmp = tf.reshape(tf.where(tf.equal(vr, g), pr, np.nan), [-1])
#             tmp = tf.gather(tmp, tf.where(tf.logical_not(tf.math.is_nan(tmp))))

#             vmeans.append(K.mean(tmp))
#             vmins.append(K.min(tmp))
#             vmaxs.append(K.max(tmp))

#         vmeans=tf.stack(vmeans)
#         vmins=tf.stack(vmins)
#         vmaxs=tf.stack(vmaxs)

#         #To avoid nans for groups with single item
#         no_scale_idx=tf.equal(vmins,vmaxs)
#         vmins=tf.gather(tf.where(no_scale_idx,0.0,vmins),vr)
#         vmaxs=tf.gather(tf.where(no_scale_idx,1.0,vmaxs),vr)

#         #Replace nan with min/2 for Convulation
#         nan_idx = tf.math.is_nan(vmeans)
#         vmeans = tf.where(nan_idx, K.min(tf.gather(vmeans, tf.where(tf.logical_not(nan_idx)))) / 2, vmeans)

#         sort_ids = tf.argsort(vmeans)
#         orig_ids = tf.argsort(sort_ids)

#         vmeans_sorted = tf.gather(vmeans, sort_ids)

#         conv_data = tf.concat([vmeans_sorted[:1] , vmeans_sorted, vmeans_sorted[-1:] * 2], axis=0)
#         conv_k = tf.ones(2, dtype=conv_data.dtype)

#         #mm lower limit
#         f0 = tf.squeeze(
#             tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 1.99, [-1, 1, 1]), 1, padding="VALID")[
#                 0][:-1])

#         #mm upper limit
#         f1 = tf.squeeze(
#             tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 2.01, [-1, 1, 1]), 1, padding="VALID")[
#                 0][1:])

#         f0=tf.gather(tf.where(no_scale_idx,0.0,tf.gather(f0, orig_ids)),vr)
#         f1=tf.gather(tf.where(no_scale_idx,1.0,tf.gather(f1, orig_ids)),vr)
        
#         ##Min Max Scaling
#         res=((pr-tf.stop_gradient(vmins))/tf.stop_gradient(vmaxs - vmins)) *  tf.stop_gradient(f1 - f0) + tf.stop_gradient(f0)
        
#         return tf.ensure_shape(res,[None,pr.shape[1]])

def group_score(pred,verts,nclusters):
    pr = pred
    vr = verts

    # vmeans,vmaxs,vmins=[],[],[]
    # for g in range(nclusters):
        
    #     tmp = np.where(vr==g,pr, np.nan).reshape(-1)
    #     tmp = tmp[np.logical_not(np.isnan(tmp))]
    #     # print(g,tmp)
    #     if len(tmp)>0:
    #         vmeans.append(np.mean(tmp))
    #         vmins.append(np.min(tmp))
    #         vmaxs.append(np.max(tmp))
    #     else:
    #         vmeans.append(np.nan)
    #         vmins.append(np.nan)
    #         vmaxs.append(np.nan)
            
            
    # vmeans=np.stack(vmeans)
    # vmins=np.stack(vmins)
    # vmaxs=np.stack(vmaxs)

    vmeans,vmaxs,vmins=np.full(nclusters,np.nan),np.full(nclusters,np.nan),np.full(nclusters,np.nan)

    for g in set(vr.tolist()):
        tmp = pr[vr==g]
        if len(tmp)>0:
            vmeans[g]=np.mean(tmp)
            vmins[g]=np.min(tmp)
            vmaxs[g]=np.max(tmp)

    #To avoid nans for groups with single item
    no_scale_idx=vmins==vmaxs
    vmins=np.where(no_scale_idx,np.asarray(0.0,dtype=vmins.dtype),vmins)[vr]
    vmaxs=np.where(no_scale_idx,np.asarray(1.0,dtype=vmaxs.dtype),vmaxs)[vr]

    


    #Replace nan with min/2 for Convulation
    nan_idx = np.isnan(vmeans)
    vmeans = np.where(nan_idx, np.min(vmeans[np.where(np.logical_not(nan_idx))]) / 2, vmeans)

    sort_ids = np.argsort(vmeans)
    orig_ids = np.argsort(sort_ids)

    vmeans_sorted = vmeans[sort_ids]

    

    conv_data = np.hstack([vmeans_sorted[:1] , vmeans_sorted, vmeans_sorted[-1:] * 2])
    conv_k = np.ones(2, dtype=conv_data.dtype)

    #mm lower limit
    f0 = np.convolve(conv_data, conv_k / 1.99, "valid")[:-1]

    #mm upper limit
    f1 = np.convolve(conv_data, conv_k / 2.01, "valid")[1:]

    f0=np.where(no_scale_idx,np.asarray(0.0,dtype=f0.dtype),f0[orig_ids])[vr]
    f1=np.where(no_scale_idx,np.asarray(1.0,dtype=f1.dtype),f1[orig_ids])[vr]
    
    ##Min Max Scaling
    res=((pr-vmins)/(vmaxs - vmins)) *  (f1 - f0) + f0
    # print("Pred Shape:",pr.shape, "GPred Shape:",res.shape, "V Shape:",vr.shape)
    return res
