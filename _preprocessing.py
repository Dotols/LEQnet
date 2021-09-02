import os
import pickle
import h5py
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import multiprocessing

data_path = os.path.join(os.path.split(os.getcwd())[0], 'data/STEAD')

def make_config(params):
    return 'W' + str(params['window_count']) + '_' + params['condition'] + '_{}'.format('imbalance' if params['imbalance'] else 'balance')

def preprocessing(params):
    if not params['imbalance']:
        t_count = 10000
        f_count = 10000
    else:
        t_count = 1000
        f_count = 9000
        
    window_size = 6000//params['window_count']
        
    if os.path.isfile(os.path.join(DIR, '../labeled_dump_STEAD', params['file_name'])):
        print('file exist!!!')
        return
    
    file_name = "chunk2.hdf5"
    csv_file = "chunk2.csv"

    # reading the csv file into a dataframe:
    df = pd.read_csv(os.path.join(data_path, csv_file))
    print(f'total events in csv file: {len(df)}')

    # filterering the dataframe
    # df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
    if params['condition'][1] == '<':
        df = df[(df.source_magnitude < int(params['condition'][2]))]
    else:
        df = df[(df.source_magnitude > int(params['condition'][2]))]
        
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    random_idx = np.random.choice(len(df),10000)
    ev_list = df['trace_name'].to_numpy()[random_idx]
    print('random choice length :', len(ev_list))

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(os.path.join(data_path, file_name), 'r')

    outputs = list()
    
    for c, evi in tqdm(enumerate(ev_list)):
        dataset = dtfl.get('data/'+str(evi)) 

        data = np.array(dataset)
        p_time_label = np.zeros(params['window_count'])
        s_time_label = np.zeros(params['window_count'])
        
        for wave in ['p', 's']:
            tmp = np.zeros(params['window_count'])
            
            idx = int(dataset.attrs['p_arrival_sample'])
            tmp[idx] = 1
            
            if params['window_count'] == 6000:
                for _ in range(1, 20):
                    try:
                        tmp[idx-_] = 1-(_*0.05)
                        tmp[idx+_] = 1-(_*0.05)
                    except:
                        pass

                if wave == 'p':
                    p_time_label = tmp
                if wave == 's':
                    s_time_label = tmp
        
        outputs.append({
            'data' : data,
            'p_label' : 1,
            's_label' : 1,
            'p_time_label' : p_time_label,
            's_time_label' : s_time_label
        })
        
        
    file_name = "chunk1.hdf5"
    csv_file = "chunk1.csv"

    # reading the csv file into a dataframe:
    df = pd.read_csv(os.path.join(data_path, csv_file))
    print(f'total events in csv file: {len(df)}')

    # filterering the dataframe
    df = df[(df.trace_category == 'noise')]
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    random_idx = np.random.choice(len(ev_list),10000)
    ev_list = df['trace_name'].to_numpy()[random_idx]
    print('random choice length :', len(ev_list))

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(os.path.join(data_path, file_name), 'r')

    for c, evi in tqdm(enumerate(ev_list)):
        dataset = dtfl.get('data/'+str(evi)) 

        data = np.array(dataset)
        p_time_label = np.zeros(params['window_count'])
        s_time_label = np.zeros(params['window_count'])

        outputs.append({
            'data' : data,
            'p_label' : 0,
            's_label' : 0,
            'p_time_label' : p_time_label,
            's_time_label' : s_time_label
        })

    with open(os.path.join(data_path, 'labeled_dump_STEAD', params['file_name']), 'wb') as f:
        pickle.dump(outputs, f)

    
from EQTransformer.core.EqT_utils import DataGenerator
from EQTransformer.core.trainer import _split, _make_dir

def preprocessing_generator(params):
    if os.path.isfile(os.path.join(data_path, 'labeled_dump_STEAD', 'train_'+params['file_name'])):
        print('file exist!!!')
        return
    
    params['input_hdf5'] = data_path+'/chunk2.hdf5'
    params['input_csv'] = data_path+'/chunk2.csv'
    params['data_size'] = int(params['total_data_size']*params['T/F_ratio'][0])
    
    print('\033[32m'+'Earthquake ' + '\033[37m' + 'preprocessing multiprocess excute!!!')
    train_eq_generator, valid_eq_generator = data_generator(params)

    params['input_hdf5'] = data_path+'/chunk1.hdf5'
    params['input_csv'] = data_path+'/chunk1.csv'
    params['data_size'] = int(params['total_data_size']*params['T/F_ratio'][1])
    
    print('\033[32m'+'Noise ' + '\033[37m' + 'preprocessing multiprocess excute!!!')
    train_n_generator, valid_n_generator = data_generator(params)

    train_data = train_eq_generator + train_n_generator     
    valid_data = valid_eq_generator + valid_n_generator 
    
    random.shuffle(train_data)
    random.shuffle(valid_data)
    
    with open(os.path.join(data_path, 'labeled_dump_STEAD', 'train_'+params['file_name']), 'wb') as f:
        pickle.dump(train_data, f)
        
    with open(os.path.join(data_path, 'labeled_dump_STEAD', 'valid_'+params['file_name']), 'wb') as f:
        pickle.dump(valid_data, f)
        
def data_generator(params):    
    params_training = {'file_name': params['input_hdf5'], 
                      'dim': params['input_dimention'][0],
                      'batch_size': params['batch_size'],
                      'n_channels': params['input_dimention'][-1],
                      'shuffle': False,  
                      'norm_mode': params['normalization_mode'],
                      'label_type': params['label_type'],               
                      'augmentation': params['augmentation'],
                      'add_event_r': params['add_event_r'], 
                      'add_gap_r': params['add_gap_r'],  
                      'shift_event_r': params['shift_event_r'],                            
                      'add_noise_r': params['add_noise_r'], 
#                       'drop_channe_r': params['drop_channel_r'],
                      'scale_amplitude_r': params['scale_amplitude_r'],
                      'pre_emphasis': params['pre_emphasis']}    

    params_validation = {'file_name': str(params['input_hdf5']),  
                         'dim': params['input_dimention'][0],
                         'batch_size': params['batch_size'],
                         'n_channels': params['input_dimention'][-1],
                         'shuffle': False,  
                         'norm_mode': params['normalization_mode'],
                         'augmentation': False}     
    

    save_dir, save_models=_make_dir(params['output_name'])
    training, validation=_split(params, save_dir)
    
    print('\033[32m'+'train ' + '\033[37m')
    train_results = multi_processing(params, DataGenerator(training, **params_training))
    print('\033[32m'+'validation ' + '\033[37m')
    valid_results = multi_processing(params, DataGenerator(validation, **params_validation))
    
    train_dataset = valid_dataset = list()
    for idx in range(params['batch_size']):
        train_dataset += train_results[idx]
        valid_dataset += valid_results[idx]
    
    train_dataset = train_dataset[:int(params['data_size']*params['train_valid_test_split'][0])]
    valid_dataset = valid_dataset[:int(params['data_size']*sum(params['train_valid_test_split'][1:]))]
        
    print('train length : \033[31m', len(train_dataset), '\033[37m / valid length : \033[31m', len(valid_dataset), '\033[37m')
    
    return train_dataset, valid_dataset
        
        
def multi_processing(params, generator):
    n_core = params['batch_size']
    window_count = params['window_count']
    if params['augmentation']: total_data_size = params['data_size']*params['train_valid_test_split'][0]
    else: total_data_size = params['data_size']*sum(params['train_valid_test_split'][1:])
        
    pool = multiprocessing.Pool(processes=n_core)
    mpc_param = [{
        'batch_number' : int(total_data_size//n_core+1),
        'window_count' : window_count,
        'core' : core, 
        'generator' : generator} 
        for core in range(n_core)]
    results = pool.map(map_item, mpc_param)
    pool.close()
    pool.join()
    
    return results

def map_item(mpc_param):
    outputs = list()
    batch_number = mpc_param['batch_number']
    window_count = mpc_param['window_count']
    generator = mpc_param['generator']
    core = mpc_param['core']
    
    prev_progress = -1
    for batch in range(batch_number):
        if core == 1:
            progress = batch*100//batch_number
            if progress != prev_progress:
                print('{}% complete'.format(progress))
            prev_progress = progress
            
        data = generator[batch]
        if window_count == 6000:
            outputs.append({
                'data' :  data[0]['input'][core],
                'p_label' :  1 if 1 in data[1]['picker_P'][core] else 0,
                's_label' :  1 if 1 in data[1]['picker_S'][core] else 0,
                'p_time_label' :  data[1]['picker_P'][core].T,
                's_time_label' :  data[1]['picker_S'][core].T
            })
        else:
            p_time_label = np.zeros((1, window_count))
            s_time_label = np.zeros((1, window_count))
            p_idx = (np.array(np.where(data[1]['picker_P'][core].T[0,:] == 1))//(6000/window_count)).astype('int32')
            s_idx = (np.array(np.where(data[1]['picker_S'][core].T[0,:] == 1))//(6000/window_count)).astype('int32')
            p_time_label[0, p_idx] = 1
            s_time_label[0, s_idx] = 1
            outputs.append({
                'data' :  data[0]['input'][core],
                'p_label' :  1 if 1 in data[1]['picker_P'][core] else 0,
                's_label' :  1 if 1 in data[1]['picker_S'][core] else 0,
                'p_time_label' : p_time_label,
                's_time_label' : s_time_label
            })
    if core == 1: 
        print('100% complete')
        
    return outputs