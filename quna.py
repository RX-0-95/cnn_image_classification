from solver import *
from cnn_model import * 
from data_utils import * 


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(subtract_mean =True)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


data_set = {
    'train_data': X_train,
    'train_label': y_train,
    'val_data':X_val,
    'val_label':y_val,
}
train_options = {
    'optimizer': 'sgd',
    'lr': 0.01,
    'epoch_num': 20,
}
#model_path = 'saved_model/trained_45'
model_path = 'tmp/trained_5'
model = tf.keras.models.load_model(model_path,compile=False)
solver= Solver(model,data_set,options=train_options)
test_loss, test_acc = solver.test_model(X_test,y_test)


model_q = Lennet5(in_channel=3, out_channel=10)

for i in range(len(model_q.trainable_variables)):
    model_q.trainable_variables[i].assign(model.trainable_variables[i])
print(model_q(X_test))
print(model(X_test))
solver= Solver(model_q,data_set,options=train_options)
test_loss, test_acc = solver.test_model(X_test,y_test)