import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from f1 import F1
from linear_decay_with_warmup import LinearDecayWithWarmup

def load_model(path):
    model = tf.keras.models.load_model(path, custom_objects={'F1': F1, 'LinearDecayWithWarmup': LinearDecayWithWarmup})
    return model

def load_dataset(filename):
    print('Reading dataset from file: {}'.format(filename))
    df = pd.read_csv(filename)
    df.info()
    if 'Class' not in df.columns:
        df['Class'] = df.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)
    training_code, testing_code = train_test_split(df[['Code', 'SATD', 'Vulnerable', 'Class']], test_size=0.1)
    training_modified = tf.data.Dataset.from_tensor_slices((training_code['Code'], training_code['SATD'], training_code['Vulnerable'], training_code['Class']))
    testing_modified = tf.data.Dataset.from_tensor_slices((testing_code['Code'], testing_code['SATD'], testing_code['Vulnerable'], testing_code['Class']))
    return training_modified, testing_modified

def load_split_dataset(data_folder):
    train = pd.read_csv(data_folder + "/train.csv")
    print('Invalid entries in train: {}'.format(train[train.isnull().any(axis=1)]))
    train.fillna('', inplace=True)
    if 'Class' not in train.columns:
        train['Class'] = train.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)

    train_mod = tf.data.Dataset.from_tensor_slices((train['Comments'], train['OnlyCode'], train['SATD'], train['Vulnerable'], train['Class']))

    val = pd.read_csv(data_folder + "/val.csv")
    val.fillna('', inplace=True)
    if 'Class' not in val.columns:
        val['Class'] = val.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)
    val_mod = tf.data.Dataset.from_tensor_slices((val['Comments'], val['OnlyCode'], val['SATD'], val['Vulnerable'], val['Class']))

    test = pd.read_csv(data_folder + "/test.csv")
    test.fillna('', inplace=True)
    if 'Class' not in test.columns:
        test['Class'] = test.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)
    test_mod = tf.data.Dataset.from_tensor_slices((test['Comments'], test['OnlyCode'], test['SATD'], test['Vulnerable'], test['Class']))
        
    print("Size of the training set: ", len(train))
    print("Size of the validation set: ", len(val))
    print("Size of the test set: ", len(test))

    return train_mod, val_mod, test_mod

def calculate_scores(predictions, label):

    if hasattr(label, "ndim") and label.ndim > 1:
        label = label.squeeze()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index in range(len(predictions)):
        prediction = predictions[index] if isinstance(predictions[index], bool) else predictions[index][0] > 0.5

        if(label[index] == True):
            if(prediction == True):
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if(prediction == False):
                tn = tn + 1
            else:
                fp = fp + 1

    print("tp -> ", tp)
    print("tn -> ", tn)
    print("fp -> ", fp)
    print("fn -> ", fn)

    precision = tp / (tp + fp) if tp + fp > 0 else -1
    recall = tp / (tp + fn) if tp + fn > 0 else -1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else -1

    print("\nprecision -> ", precision)
    print("recall -> ", recall)
    print("accuracy -> ", accuracy)
    print("f1 -> ", f1)

def generate_path(output_dir, model_name, learning_rate, number_of_epochs, batch_size, dropout_prob, l2_reg_lambda):
    return "{}/weights_{}_lr_{}_ne_{}_bs_{}_dp_{}_l2_{}.tf".format(output_dir, model_name, learning_rate, number_of_epochs, batch_size, dropout_prob, l2_reg_lambda)