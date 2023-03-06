import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef
import energyflow.archs
from energyflow.archs import PFN
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer, concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization

plt.rc('font', size=20)

#These are the same datasets from the OmniFold paper https://arxiv.org/abs/1911.09107.  More detail at https://energyflow.network/docs/datasets/.
#Pythia and Herwig are two generators; one will be treated here as the "simulation" and one as "data".
datasets = {'Pythia26': ef.zjets_delphes.load('Pythia26', num_data=1000000),
            'Herwig': ef.zjets_delphes.load('Herwig', num_data=1000000)}
datasets = {'Pythia26': ef.zjets_delphes.load('Pythia26', num_data=1000000),
            'Herwig': ef.zjets_delphes.load('Herwig', num_data=1000000)}

def is_charged(myin):
    if (myin == 0):
        return 0
    elif (myin == 0.1):
        return 1
    elif (myin == 0.2):
        return -1
    elif (myin == 0.3):
        return 0
    elif (myin == 0.4):
        return -1
    elif (myin == 0.5):
        return 1
    elif (myin == 0.6):
        return -1
    elif (myin == 0.7):
        return 1
    elif (myin == 0.8):
        return 1
    elif (myin == 0.9):
        return -1
    elif (myin == 1.0):
        return 1
    elif (myin == 1.1):
        return -1
    elif (myin == 1.2):
        return 0
    elif (myin == 1.3):
        return 0
    


for dataset in datasets:
    mycharges = []
    mycharges2 = []
    for i in range(len(datasets[dataset]['gen_particles'])):
        pTs = datasets[dataset]['gen_particles'][i][:,0]
        charges = [is_charged(datasets[dataset]['gen_particles'][i][:,3][j]) for j in range(len(datasets[dataset]['gen_particles'][i][:,3]))]
        mycharges+=[np.sum(charges*pTs**0.5)/np.sum(pTs**0.5)]
        mycharges2+=[np.sum(np.abs(charges)*pTs)/np.sum(pTs)]
    datasets[dataset]['gen_charge'] = mycharges
    datasets[dataset]['gen_pTcharge'] = mycharges2

    mycharges = []
    mycharges2 = []
    for i in range(len(datasets[dataset]['sim_particles'])):
        pTs = datasets[dataset]['sim_particles'][i][:,0]
        charges = [is_charged(datasets[dataset]['sim_particles'][i][:,3][j]) for j in range(len(datasets[dataset]['sim_particles'][i][:,3]))]
        mycharges+=[np.sum(charges*pTs**0.5)/np.sum(pTs**0.5)]
        mycharges2+=[np.sum(np.abs(charges)*pTs)/np.sum(pTs)]
    datasets[dataset]['sim_charge'] = mycharges
    datasets[dataset]['sim_pTcharge'] = mycharges2
    
tau2s_Pythia_sim = datasets['Pythia26']['sim_tau2s']
tau2s_Herwig_sim = datasets['Herwig']['sim_tau2s']

tau1s_Pythia_sim = datasets['Pythia26']['sim_widths']
tau1s_Herwig_sim = datasets['Herwig']['sim_widths']

tau2s_Pythia_gen = datasets['Pythia26']['gen_tau2s']
tau2s_Herwig_gen = datasets['Herwig']['gen_tau2s']

tau1s_Pythia_gen = datasets['Pythia26']['gen_widths']
tau1s_Herwig_gen = datasets['Herwig']['gen_widths']

pT_true = datasets['Pythia26']['gen_jets'][:,0]
m_true = datasets['Pythia26']['gen_jets'][:,3]
pT_reco = datasets['Pythia26']['sim_jets'][:,0]
m_reco = datasets['Pythia26']['sim_jets'][:,3]

pT_true_alt = datasets['Herwig']['gen_jets'][:,0]
m_true_alt = datasets['Herwig']['gen_jets'][:,3]
pT_reco_alt = datasets['Herwig']['sim_jets'][:,0]
m_reco_alt = datasets['Herwig']['sim_jets'][:,3]

#
w_true = datasets['Pythia26']['gen_widths']
w_reco = datasets['Pythia26']['sim_widths']
w_true_alt = datasets['Herwig']['gen_widths']
w_reco_alt = datasets['Herwig']['sim_widths']

#
q_true = np.array(datasets['Pythia26']['gen_charge'])
q_reco = np.array(datasets['Pythia26']['sim_charge'])
q_true_alt = np.array(datasets['Herwig']['gen_charge'])
q_reco_alt = np.array(datasets['Herwig']['sim_charge'])

#
r_true = np.array(datasets['Pythia26']['gen_pTcharge'])
r_reco = np.array(datasets['Pythia26']['sim_pTcharge'])
r_true_alt = np.array(datasets['Herwig']['gen_pTcharge'])
r_reco_alt = np.array(datasets['Herwig']['sim_pTcharge'])

gauss_data = np.random.normal(0,1,100000)
gauss_sim = np.random.normal(-0.5,1,100000)

initializer = tf.keras.initializers.RandomUniform(minval=-5., maxval=5.)
n = 1

class MyLayer(Layer):

    def __init__(self, myc, **kwargs):
        self.myinit = myc
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._lambda = self.add_weight(name='lambda', 
                                    shape=(n,),
                                    initializer=tf.keras.initializers.Constant(self.myinit), 
                                    trainable=True)
        
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #return tf.exp(self._lambda1 * x + self._lambda0)
        return tf.exp(sum([self._lambda[i]* x**(i+1) for i in range(n)]))
    
def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    weights_1 = K.sum(y_true*weights)
    weights_0 = K.sum((1-y_true)*weights)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred)/weights_1 +
                         (1 - y_true) * K.log(1 - y_pred)/weights_0)
    return K.mean(t_loss)

def weighted_binary_crossentropy_GAN(y_true, y_pred):
    weights = tf.gather(y_pred, [1], axis=1) # event weights
    y_pred = tf.gather(y_pred, [0], axis=1) # actual y_pred for loss

    weights_1 = K.sum(y_true*weights)
    weights_0 = K.sum((1-y_true)*weights)

    #tf.print("weights",weights_0,weights_1)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = weights * ((1 - y_true) * K.log(1 - y_pred)/weights_0)
    return K.mean(t_loss)

#xvals_1 = np.concatenate([gauss_data,gauss_sim])
#yvals_1 = np.concatenate([np.ones(len(gauss_data)),np.zeros(len(gauss_sim))])
N = 20
n = 1
data_moments = np.zeros((N, N))
unweighted_moments = np.zeros((N, N))
weighted_moments = np.zeros((N, N))

while n <= N:
    print(f"{n = }")
    myc = 0.1
    mymodel_inputtest = Input(shape=(1,))
    mymodel_test = MyLayer(myc)(mymodel_inputtest)
    model_generator = Model(mymodel_inputtest, mymodel_test)

    inputs_disc = Input((1, ))
    hidden_layer_1_disc = Dense(50, activation='relu')(inputs_disc)
    hidden_layer_2_disc = Dense(50, activation='relu')(hidden_layer_1_disc)
    hidden_layer_3_disc = Dense(50, activation='relu')(hidden_layer_2_disc)
    outputs_disc = Dense(1, activation='sigmoid')(hidden_layer_3_disc)
    model_discrimantor = Model(inputs=inputs_disc, outputs=outputs_disc)
    
    model_discrimantor.compile(loss=weighted_binary_crossentropy, optimizer='adam')

    model_discrimantor.trainable = False
    mymodel_gan = Input(shape=(1,))
    gan_model = Model(inputs=mymodel_gan,outputs=concatenate([model_discrimantor(mymodel_gan),model_generator(mymodel_gan)]))

    gan_model.compile(loss=weighted_binary_crossentropy_GAN, optimizer='adam')

    xvals_1 = np.concatenate([w_true_alt,w_true])
    yvals_1 = np.concatenate([np.ones(len(w_true_alt)),np.zeros(len(w_true))])

    X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(xvals_1, yvals_1)

    n_epochs = int(np.ceil(np.linspace(20, 100, N)[n]))
    n_batch = 128*10
    n_batches = len(X_train_1) // n_batch

    for i in range(n_epochs):
        #print("  ",np.sum(model_generator.predict(X_train_1,batch_size=1000)))
        for j in range(n_batches):
            X_batch = X_train_1[j*n_batch:(j+1)*n_batch]
            Y_batch = Y_train_1[j*n_batch:(j+1)*n_batch]
            W_batch = model_generator(X_batch)
            W_batch = np.array(W_batch).flatten()
            W_batch[Y_batch==1] = 1
            #W_batch[Y_batch==0] = 1

            Y_batch_2 = np.stack((Y_batch, W_batch), axis=1)

            model_discrimantor.train_on_batch(X_batch, Y_batch_2)

            gan_model.train_on_batch(X_batch[Y_batch==0],np.zeros(len(X_batch[Y_batch==0])))
            
        mylambda = np.array(model_generator.layers[-1].get_weights())
        print("on epoch=",i, mylambda)


    xvals_1 = np.concatenate([w_true_alt,w_true])
    yvals_1 = np.concatenate([np.ones(len(w_true_alt)),np.zeros(len(w_true))])
    arr = np.array([mylambda[:, k]*w_true**(k+1) for k in range(n)])
    exponent = np.exp(np.sum(arr, axis=0))
    weights_1 = np.concatenate([np.ones(len(w_true_alt)),exponent*len(w_true_alt)/np.sum(exponent)])

    X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

    #data, simulation w/o weights, weighted simulation
    data_moments[:, n-1] = [np.mean(X_test_1[Y_test_1==1]**(i+1)) for i in range(N)]
    unweighted_moments[:, n-1] = [np.mean(X_test_1[Y_test_1==0]**(i+1)) for i in range(N)]
    weighted_moments[:, n-1] = [np.average(X_test_1[Y_test_1==0]**(i+1),weights=w_test_1[Y_test_1==0]) for i in range(N)]
    n += 1
    print("\n\n\n")
    
print(f"{data_moments = }")
print(f"{unweighted_moments = }")
print(f"{weighted_moments = }")