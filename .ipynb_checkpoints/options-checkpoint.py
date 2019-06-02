# Arguments for restaurant,AOA
from argparse import Namespace


dataset_files = {
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
}
input_colses = {
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'aen': ['text_raw_indices', 'aspect_indices'],
        'aen_bert' : ['text_raw_bert_indices', 'aspect_bert_indices']
    }

AOA_opt = Namespace(
    model_name="aoa",
    dataset='restaurant',#twitter,laptop
    seed=1234,
    optimizer = 'adam',
    initializer = 'xavier_uniform_',
    log_step = 5,
    logdir = 'log',
    embed_dim = 200,
    hidden_dim = 300,
    max_seq_len = 80,
    polarities_dim = 3,
    hops = 3,
    device = None,
    learning_rate = 0.001,
    batch_size = 128,
    l2reg = 0.0001,#0.00001
    num_epoch = 20,
    dropout = 0.2,
    inputs_cols = input_colses['aoa'],
    dataset_file = dataset_files['restaurant']
)

AEAT_LSTM_opt = Namespace(
    model_name="atae_lstm",
    dataset='restaurant',#twitter,laptop
    seed=1234,
    optimizer = 'adam',
    initializer = 'xavier_uniform_',
    log_step = 5,
    logdir = 'log',
    embed_dim = 200,
    hidden_dim = 300,
    max_seq_len = 80,
    polarities_dim = 3,
    hops = 3,
    device = None,
    learning_rate = 0.001,
    batch_size = 128,
    l2reg = 0.00001,
    num_epoch = 20,
    dropout = 0,
    inputs_cols = input_colses['atae_lstm'],
    dataset_file = dataset_files['restaurant']
)

AEN_opt = Namespace(
    model_name="aen",
    dataset='restaurant',#twitter,laptop
    seed=1234,
    optimizer = 'adam',
    initializer = 'xavier_uniform_',
    log_step = 5,
    logdir = 'log',
    embed_dim = 300,
    hidden_dim = 300,
    max_seq_len = 80,
    polarities_dim = 3,
    hops = 3,
    device = None,
    learning_rate = 0.002,
    batch_size = 128,
    l2reg = 0.00001,
    num_epoch = 20,
    dropout = 0.1,
    inputs_cols = input_colses['aen'],
    dataset_file = dataset_files['restaurant']
)

AEN_BERT_opt = Namespace(
    model_name="aen_bert",
    dataset='restaurant',#twitter,laptop
    seed=1234,
    optimizer = 'adam',
    initializer = 'xavier_uniform_',
    log_step = 5,
    logdir = 'log',
    embed_dim = 200,
    hidden_dim = 300,
    max_seq_len = 80,
    polarities_dim = 3,
    hops = 3,
    device = None,
    learning_rate = 0.001,
    batch_size = 128,
    l2reg = 0.00001,
    num_epoch = 20,
    dropout = 0,
    inputs_cols = input_colses['aen_bert'],
    dataset_file = dataset_files['restaurant']
)


