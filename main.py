import time
import os
import yaml
import io
import numpy as np
from argparse import ArgumentParser
from keras_tuner import Objective
from keras.callbacks import ModelCheckpoint

from model_strategy import ModelStrategy
from utils import load_split_dataset, load_model, generate_path
from hyper_model import HyperModelFactoryAdapter
from no_check_gridsearch import NoCheckGridSearch
from custom_multitask_metric_callback import CustomMultitaskMetricCallback


def train_model(model_name,
                separate_comments,
                mask_satd_keywords,
                truncation_side,
                learning_rate,
                number_of_epochs,
                batch_size,
                dropout_prob,
                l2_reg_lambda,
                shared_layer,
                training_data,
                validation_data,
                store_weights,
                output_dir,
                limit,
                gamma):
    
    model_strategy = ModelStrategy(model_name)
    encoder = model_strategy.create_encoder(separate_comments, mask_satd_keywords, truncation_side)
    ds_train_encoded, y_train, ds_validation_encoded, y_validation = encoder.encode_train_and_validation_data(training_data, validation_data, limit)

    train_size = len(y_train['satd']) if model_name == 'multitask' else len(y_train)
    steps_per_epoch = train_size // batch_size + 1 
    decay_steps = steps_per_epoch * number_of_epochs
    warmup_steps = decay_steps // 5
    model = model_strategy.create_model(learning_rate, dropout_prob, l2_reg_lambda, shared_layer, warmup_steps, decay_steps, gamma)

    print('Training {}'.format(model_name))

    callbacks = [CustomMultitaskMetricCallback()] if model_name == 'multitask' else []

    if store_weights:
        path = generate_path(output_dir, model_name, learning_rate, number_of_epochs, batch_size, dropout_prob, l2_reg_lambda)
        callbacks.append(ModelCheckpoint(filepath=path, 
                                         save_best_only=True, 
                                         monitor=model_strategy.get_metric_to_monitor(), 
                                         mode=model_strategy.get_metric_direction()))

    model.fit(ds_train_encoded, 
              y_train,
              epochs=number_of_epochs, 
              batch_size=batch_size, 
              validation_data=(ds_validation_encoded, y_validation),
              callbacks = callbacks)


def test_saved_model(model_name, model_file, separate_comments, mask_satd_keywords, truncation_side, test_data, limit):
    
    print('Testing saved model {}'.format(model_name))

    model = load_model(model_file)

    model_factory = ModelStrategy(model_name)
    encoder = model_factory.create_encoder(separate_comments, mask_satd_keywords, truncation_side)
    
    ds_test_encoded, test_labels = encoder.encode_examples(test_data, limit)
    predictions = model.predict(ds_test_encoded)

    print("Measures for {}\n".format(model_name))

    model_factory.create_scores_calculator().calculate(test_labels, predictions)

def test_combination_saved_models(model_file_satd, separate_comments, mask_satd_keywords, truncation_side, test_data, limit):
    
    print('Testing combination of saved models')

    model_satd = load_model(model_file_satd)

    model_factory = ModelStrategy('vulsatd')
    encoder = model_factory.create_encoder(separate_comments, mask_satd_keywords, truncation_side)
    
    ds_test_encoded, test_labels = encoder.encode_examples(test_data, limit)
    satd_predictions = model_satd.predict(ds_test_encoded)

    model_vul = load_model(model_file_satd.replace('satdonly', 'vulonly'))
    vul_predictions = model_vul.predict(ds_test_encoded)

    satd_predictions_boolean = satd_predictions > 0.5
    vul_predictions_boolean = vul_predictions > 0.5

    vulsatd_prediction = np.logical_and(satd_predictions_boolean, vul_predictions_boolean)

    print("Measures for VulSATD\n")

    model_factory.create_scores_calculator().calculate(test_labels, vulsatd_prediction)    

def run_hyper_analysis(model_name, number_of_epochs, config, training_data, validation_data, store_weights, output_dir):

    model_strategy = ModelStrategy(model_name)

    encoder = model_strategy.create_encoder()

    ds_train_encoded, y_train, ds_validation_encoded, y_validation = encoder.encode_train_and_validation_data(training_data, validation_data)

    hyper_model = HyperModelFactoryAdapter(model_strategy,
                                           lambda hp: hp.Choice('learning_rate', config['learning_rate']),
                                           lambda hp: hp.Choice('dropout_prob', config['dropout_prob']),
                                           lambda hp: hp.Choice('l2_reg_lambda', config['l2_reg_lambda']),
                                           lambda hp: hp.Choice('shared_layer', config['shared_layer']),
                                           lambda hp: hp.Choice('batch_size', config['batch_size']))

    metric_to_monitor = model_strategy.get_metric_to_monitor()
    metric_direction = model_strategy.get_metric_direction()

    tuner = NoCheckGridSearch(hyper_model, 
                              objective=Objective(metric_to_monitor, direction=metric_direction), 
                              executions_per_trial=1, 
                              overwrite=True, 
                              directory=output_dir)

    tuner.search_space_summary()

    callbacks = [CustomMultitaskMetricCallback()] if model_name == 'multitask' else []

    if store_weights:
        callbacks.append(ModelCheckpoint(filepath=os.path.join(output_dir, model_name), 
                                         save_best_only=True, 
                                         monitor=metric_to_monitor, 
                                         mode=metric_direction))

    tuner.search(ds_train_encoded, 
                 y_train,
                 epochs=number_of_epochs, 
                 validation_data=(ds_validation_encoded, y_validation),
                 callbacks=callbacks)
    
    tuner.results_summary()

def str2bool(v):
    parser = ArgumentParser()
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise parser.error('Boolean value expected.')

def read_args():

    parser = ArgumentParser()

    parser.add_argument('--model', choices=['satdonly', 'vulonly', 'vulsatd', 'multitask'], required=True)
    parser.add_argument('--mode', choices=['hyper-analysis', 'train', 'test', 'test-combination'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--separate-comments', type=str2bool, default=True)
    parser.add_argument('--mask-satd-keywords', choices=['none', 'mask', 'remove'], default='none')
    parser.add_argument('--truncation-side', choices=['left', 'right'], default='right')
    parser.add_argument('--shared-layer', type=str2bool, default=True)
    parser.add_argument('--model-file', type=str, default=None,
                        help='In test mode, the file to load the weights from.')
    parser.add_argument('--store-weights', type=bool, default=False,
                        help='If the weights should be saved in a file.')
    parser.add_argument('--output-dir', type=str, default='stored_models',
                        help='The output directory to save weights if store-weights is enabled.')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dropout-prob', type=float, default=0.1)
    parser.add_argument('--l2-reg-lambda', type=float, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--gamma', type=float, required=False)

    return parser.parse_args()

if __name__ == "__main__":
    
    args = read_args()

    training_data, validation_data, test_data = load_split_dataset(args.dataset)

    start_time = time.time()

    if args.mode == 'hyper-analysis':
        with io.open('hyperanalysis.yml', 'r') as config_file:
            config = yaml.safe_load(config_file)
        run_hyper_analysis(args.model, 
                           args.epochs,
                           config,
                           training_data, 
                           validation_data, 
                           args.store_weights, 
                           args.output_dir)
    elif args.mode == "test":
        test_saved_model(args.model, 
                         args.model_file, 
                         args.separate_comments, 
                         args.mask_satd_keywords, 
                         args.truncation_side, 
                         test_data,
                         args.limit)
    elif args.mode == "test-combination":
        test_combination_saved_models(args.model_file,
                                      args.separate_comments,
                                      args.mask_satd_keywords,
                                      args.truncation_side,
                                      test_data,
                                      args.limit)
    elif args.mode == "train":
        train_model(args.model,
                    args.separate_comments,
                    args.mask_satd_keywords,
                    args.truncation_side,
                    args.learning_rate,
                    args.epochs,
                    args.batch_size,
                    args.dropout_prob,
                    args.l2_reg_lambda,
                    args.shared_layer,
                    training_data,          
                    validation_data,            
                    args.store_weights,
                    args.output_dir,
                    args.limit,
                    args.gamma)

    print("--- %s seconds taken to run ---" % (time.time() - start_time))
