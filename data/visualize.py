import numpy as np
import pickle

train_test_split = './my_full_2dofa_scannet.pkl' #'./my_scannet_standard_train_test_val_split.pkl' #'scannet_standard_train_test_val_split.pkl' #''./rectified_2dofa_scannet.pkl' #'
data_info = pickle.load(open(train_test_split, 'rb'))

if train_test_split == './full_2dofa_scannet.pkl' or train_test_split == './my_full_2dofa_scannet.pkl':
    train_data = data_info['train']
    test_data = data_info['test']
    print('with_ga[e2] = ', len(train_data['with_ga']['e2'][0]))
    print('with_ga[-e2] = ', len(train_data['with_ga']['-e2'][0]))
    print('no_ga[e2] = ', len(train_data['no_ga'][0]))
elif train_test_split == './my_scannet_standard_train_test_val_split.pkl' or  train_test_split == './scannet_standard_train_test_val_split.pkl':
    train_data = data_info['train']
    test_data = data_info['test']
    val_data = data_info['val']
    print('train_data = ', len(train_data[0]))
    print('val_data = ', len(val_data[0]))
    print('test_data = ', len(test_data[0]))
    print('test sample 0 = ', test_data[0][0])
elif train_test_split == './rectified_2dofa_scannet.pkl':
    train_data = data_info['train']
    test_data = data_info['test']
    val_data = data_info['val']
    print('train_data [e2] = ', len(train_data['e2']))
    print('train_data [-e2] = ', len(train_data['-e2']))
    print('val_data [e2] = ', len(val_data['e2']))
    print('val_data [-e2] = ', len(val_data['-e2']))
    print('test_data [e2] = ', len(test_data['e2']))
    print('test_data [-e2] = ', len(test_data['-e2']))

exit(0)
max = 100
for sample in val_data[0]:#[mode]:
    file_id = int(sample[-16:-10])
    #subdir_idx = int(sample[-30:-26])
    #print(sample[-30:-26])
    if file_id == 0:
        print('0 detected {}'.format(sample))





