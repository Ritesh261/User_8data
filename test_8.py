import numpy as np
#from utils8_bf import *
#from kaggle_utils import *
from tensorflow.python.keras.layers import *
from matplotlib import pyplot as plt

# ------------------------------------------
#  Load and generate simulation data
# ------------------------------------------
#path = '/home/manchikanti/Work/Final_year_Project/my_data_sets/Train'  # the path of the dictionary containing pcsi.mat and ecsi.mat
#path = 'D:\\work\\3rd year\\6th sem\\capestone\\User_2\\07-05-2021\\data_set_23_04\\0db\\example\\Test'
#path = 'D:\work\3rd year\6th sem\capestone\User_2\train_set\example\Test'
path = '../input/user-8-our/-20db/Test'
H_1, H_1_est, H_2, H_2_est, H_3, H_3_est, H_4, H_4_est, H_5, H_5_est, H_6, H_6_est, H_7, H_7_est, H_8, H_8_est = mat_load(path)
H = np.concatenate((H_1,H_2,H_3,H_4,H_5,H_6,H_7,H_8), axis = 2)
H = H[range(0,1000),:]
H_est = np.concatenate((H_1_est,H_2_est,H_3_est,H_4_est,H_5_est,H_6_est,H_7_est,H_8_est), axis = 2)
H_est = H_est[range(0,1000),:]


# use the estimated csi as the input of the BFNN
H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
# H denotes the perfect csi
#H = np.squeeze(H)
H = np.resize(H,(1000,512))
H_est = np.resize(H_est,(1000,512))
# generate  SNRs associated with different samples
SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)

# -----------------------
#  Construct the BFNN Model
# -----------------------
# imperfect CSI is used to output the vrf
imperfect_CSI = Input(name='imperfect_CSI', shape=(H_input.shape[1:4]), dtype=tf.float32)
# perfect_CSI is only used to compute the loss, and not required in prediction
perfect_CSI = Input(name='perfect_CSI', shape=(H.shape[1],), dtype=tf.complex64)
# the SNR is also fed into the BFNN
SNR_input = Input(name='SNR_input', shape=(1,), dtype=tf.float32)
temp = BatchNormalization()(imperfect_CSI)
temp = Flatten()(temp)
temp = BatchNormalization()(temp)
temp = Dense(2048, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(1024, activation='relu')(temp)
phase = Dense(8*Nt)(temp)
V_RF = Lambda(trans_Vrf, dtype=tf.complex64, output_shape=(Nt,))(phase)
rate = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, V_RF, SNR_input])
#power = Lambda(power_func, dtype=tf.complex64, output_shape=(8*Nt,))([V_RF, SNR_input])
#model = Model(inputs=[imperfect_CSI, perfect_CSI, SNR_input], outputs=power)
model = Model(inputs=[imperfect_CSI, perfect_CSI, SNR_input], outputs=rate)
# the y_pred is the actual rate, thus the loss is y_pred, without labels
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
model.summary()

# -----------------------
#  Test Your Model
# -----------------------
batch = batchsize
rate = []
# load the trained model
# You can train the model by train.py or download it from the Google drive provided.
#model.load_weights('F:\google_driver\BFNN\data sets/20db./20db.h5')
#model.load_weights('data sets/20db/20db.h5')
#model.load_weights('data sets/0db/0db.h5')
#model.load_weights('data sets/-20db/-20db.h5')
model.load_weights('./-20db8trial.h5')
#model.load_weights('./temp_trained.h5')

for snr in range(-20, 25, 5):
    SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
    y = model.evaluate(x=[H_input, H, SNR], y=H, batch_size=batch)
    rate.append(-y)
print(rate)


plt.title("The result of BFNN training with 20db")
plt.xlabel("SNR(dB)")
plt.ylabel("Spectral Efficiency")
plt.plot(range(-20, 25, 5), rate)
plt.show()
'''
snr = -20
SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
y = model.evaluate(x=[H_input, H, SNR], y=H, batch_size=10000)
print(y)
y_pred = model.predict(x=[H_input, H, SNR])
y_pred_1, y_pred_2 = tf.split(y_pred[0], num_or_size_splits = 2, axis = 0, name = 'split')
y_pred_final = y_pred_1 + y_pred_2
print(y_pred_final)
plt.plot(y_pred_final)
plt.show()
'''	