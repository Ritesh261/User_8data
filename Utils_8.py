from tensorflow.python.keras import *
import tensorflow as tf
import scipy.io as sio
import numpy as np
# ---------------------
#  Global Parameters
# ---------------------
Nt = 64  # the number of antennas
P = 1   # the normalized transmit power
																								

# ---------------------
#  Functions
# ---------------------

# transfer the phase to complex-valued analog beamformer
def trans_Vrf(temp):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

batchsize = int(input("please enter the batchsize you want to run the model"))
print("the batch size of the model is -->" + str(batchsize))
# For the simplification of implementation based on Keras, we use a lambda layer to compute the rate
# Thus, the output of the model is actually the loss.
def Rate_func(temp):
    #h, v, SNR_input, batchsize = temp
    h, v, SNR_input = temp
    #here we split the hte h and the V_rf we get for getting individual users
    print("\n the shape of the h is----- ")
    print(h.shape)
    #batchsize = 50
    h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8 = tf.split(h, num_or_size_splits = 8, axis = 1, name = 'split')
    v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8 = tf.split(v, num_or_size_splits = 8, axis = 1, name = 'split')
    print("\n the shape of the h_1")
    print(h_1.shape)
    print("\n the shape of the v_1")
    print(v_1.shape)
    H =tf.concat((h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8 ), 0)
    print("\n the shape of the H cap in rate func")
    print(H.shape)
    V =tf.transpose(tf.concat((v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8), 0))
    print("\n the shape of the v in rate func")
    print(V.shape)
    mat = tf.linalg.matmul(H,V)
    row = tf.zeros([1,8*batchsize], dtype = tf.complex64)
    for i in range(0,(8*batchsize),8):
        if(i == 0):
            z_2 = tf.zeros([8,(8*batchsize)-8], dtype = tf.complex64)
            ele = tf.gather(tf.gather(mat,indices = range(0,8)), indices = range(0,8), axis = 1)
            row_gen = tf.concat([ele,z_2], axis = 1)
        elif(i == ((8*batchsize)-8)):
            z_1 = tf.zeros([8,(8*batchsize)-8], dtype = tf.complex64)
            ele = tf.gather(tf.gather(mat,indices = range(i,i+8)), indices = range(i,i+8), axis = 1)
            row_gen = tf.concat([z_1,ele], axis = 1)
        else:
            z_1 = tf.zeros([8,i], dtype = tf.complex64)
            z_2 = tf.zeros([8,(8*batchsize - (i+8))], dtype = tf.complex64)
            ele = tf.gather(tf.gather(mat,indices = range(i,i+8)), indices = range(i,i+8), axis = 1)
            row_gen = tf.concat([z_1,ele,z_2], axis = 1) 
        row = tf.concat([row,row_gen],0)
        print("the shape of the row is --",row.shape)
    indices = range(1,(8*batchsize+1))
    mat = tf.gather(row, indices, axis = 0)
    print("the shape of the mat is --",mat.shape)
    v_d = tf.linalg.inv(mat)        
    print("\n the shape of the V_d in rate func")
    print(v_d.shape)
    #print(v_d)
    X = tf.linalg.matmul(V,v_d)
    print("\n the shape of the X in rate func")
    print(X.shape)
    arr = tf.zeros([1,1], dtype = tf.float32)
    j = 0
    for i in range(0,(8*batchsize),8):
        op_0 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i], axis=1))),2))
        op_1 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i +1], axis=1))),2))
        op_2 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 2], axis=1))),2))
        op_3 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 3], axis=1))),2))
        op_4 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 4], axis=1))),2))
        op_5 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 5], axis=1))),2))
        op_6 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 6], axis=1))),2))
        op_7 = 1/((8)*tf.pow(tf.norm(tf.abs(tf.gather(X, [i + 7], axis=1))),2))
        snr = tf.gather(SNR_input,[j],axis = 0)
        op_0 = tf.math.log(1+(snr*op_0))/tf.math.log(2.0)
        op_1 = tf.math.log(1+(snr*op_1))/tf.math.log(2.0)
        op_2 = tf.math.log(1+(snr*op_2))/tf.math.log(2.0)
        op_3 = tf.math.log(1+(snr*op_3))/tf.math.log(2.0)
        op_4 = tf.math.log(1+(snr*op_4))/tf.math.log(2.0)
        op_5 = tf.math.log(1+(snr*op_5))/tf.math.log(2.0)
        op_6 = tf.math.log(1+(snr*op_6))/tf.math.log(2.0)
        op_7 = tf.math.log(1+(snr*op_7))/tf.math.log(2.0)
        op = op_0 + op_1 + op_2 + op_3 + op_4 + op_5 + op_6 + op_7
        j += 1
        arr = tf.concat([arr,op], axis = 0)
    j =0
    print("\n the shape of the array is")
    print(arr.shape)
    indices = range(1,batchsize+1)
    P_mat = tf.gather(arr, indices, axis=0)
    print("#########################")
    print("the shape of the P_mat is ")
    print(P_mat.shape)
    #rate_new = (tf.cast(tf.reduce_sum(P_mat, axis = 1),tf.float32))
    rate_new = P_mat
    print("the rate_new")
    print(rate_new.shape)
    #print(rate_new)
    return -rate_new

def power_func(temp):
	v, SNR_input = temp
	print("------------------------------------",v)
	print("-------------------------------------",SNR_input)
	#vSNR_imput = tf.keras.backend.batch_dot(, SNR_input)
	power = (v*tf.cast(SNR_input,tf.complex64)*tf.cast(tf.math.sqrt(2.), tf.complex64))/(2*Nt)
	power=tf.pow((tf.abs(power)),2)
	print("--------------------------------",power)
	#return power
	return power
	
# load the saved .mat files generated by Matlab.
#path = 'C:/Users/Vaishnavi/Desktop/Beamforming_2users_W02/train_set/example/Train'
#path = 'C:/Users/Vaishnavi/Desktop/Bf_revised/train_set/example/Train'


def mat_load(path):
    print('loading data...')
    # load the perfect csi
    h_1 = sio.loadmat(path + '/H_1.mat')['val']
    # load the estimated csi
    h_1_est = sio.loadmat(path + '/H_1_est.mat')['val']
    # load the perfect csi for the second user
    h_2 = sio.loadmat(path + '/H_2.mat')['val']
    #load the estimated csi for the second user
    h_2_est = sio.loadmat(path + '/H_2_est.mat')['val']

    h_3 = sio.loadmat(path + '/H_3.mat')['val']
    h_3_est = sio.loadmat(path + '/H_3_est.mat')['val']

    h_4 = sio.loadmat(path + '/H_4.mat')['val']
    h_4_est = sio.loadmat(path + '/H_4_est.mat')['val']

    h_5 = sio.loadmat(path + '/H_5.mat')['val']
    h_5_est = sio.loadmat(path + '/H_5_est.mat')['val']

    h_6 = sio.loadmat(path + '/H_6.mat')['val']
    h_6_est = sio.loadmat(path + '/H_6_est.mat')['val']

    h_7 = sio.loadmat(path + '/H_7.mat')['val']
    h_7_est = sio.loadmat(path + '/H_7_est.mat')['val']

    h_8 = sio.loadmat(path + '/H_8.mat')['val']
    h_8_est = sio.loadmat(path + '/H_8_est.mat')['val']



    print('loading complete')
    #print('The shape of CSI is: ', h_1_est.shape)
    return h_1, h_1_est, h_2, h_2_est, h_3, h_3_est, h_4, h_4_est, h_5, h_5_est, h_6, h_6_est, h_7, h_7_est, h_8, h_8_est