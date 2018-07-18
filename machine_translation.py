#Arnav Bansal
import os
import pickle
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

def load_data(path):
    input_file = os.path.join(path)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()
    
def create_lookup_tables(text):
    CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)
    
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i
    
    return vocab_to_int

def preprocess(source_path, target_path):
    source_text = load_data(source_path).lower()
    target_text = load_data(target_path).lower()
    source_vocab_to_int = create_lookup_tables(source_text)
    target_vocab_to_int = create_lookup_tables(target_text)
    source_id_text = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int[word] for word in sentence.split()] + [target_vocab_to_int['<EOS>']] for sentence in target_text.split('\n')]
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump((source_id_text, target_id_text, source_vocab_to_int, target_vocab_to_int), out_file)

def model_inputs():
    input_ = tf.placeholder(tf.int32, [None, None], 'input')
    targets = tf.placeholder(tf.int32, [None, None])
    learn_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_seq_len = tf.placeholder(tf.int32, [None], 'target_sequence_length')
    max_target_seq_len = tf.reduce_max(target_seq_len, name='max_target_len')
    source_seq_len = tf.placeholder(tf.int32, [None], 'source_sequence_length')
    return (input_, targets, learn_rate, keep_prob, target_seq_len, max_target_seq_len, source_seq_len)

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    return tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), target_data[:, :-1]], 1)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, source_sequence_length, source_vocab_size, encoding_embedding_size):
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    
    def make_cell(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, 2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    return tf.nn.dynamic_rnn(cell, embed, source_sequence_length, dtype=tf.float32)

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length, output_layer, keep_prob):
    training_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)
    return tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True, maximum_iterations=max_summary_length)[0]

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob):
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size])
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)
    return tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)[0]

def decoding_layer(dec_input, encoder_state, target_sequence_length, max_target_sequence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, decoding_embedding_size):
    embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    def make_cell(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, 2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    with tf.variable_scope('decode') as decode_scope:
        training_decoder_outputs = decoding_layer_train(encoder_state, cell, embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)
        decode_scope.reuse_variables()
        inference_decoder_outputs = decoding_layer_infer(encoder_state, cell, embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], max_target_sequence_length, target_vocab_size, output_layer, batch_size, keep_prob)
    
    return training_decoder_outputs, inference_decoder_outputs

def seq2seq_model(input_data, target_data, keep_prob, batch_size, source_sequence_length, target_sequence_length, max_target_sentence_length, source_vocab_size, target_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    _, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_sequence_length, source_vocab_size, enc_embedding_size)
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    return decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)

epochs = 10
batch_size = 256
rnn_size = 256
num_layers = 2
encoding_embedding_size = 128
decoding_embedding_size = 128
learning_rate = 0.001
keep_probability = 0.5
display_step = 20

save_path = 'checkpoints/dev'
source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'

preprocess(source_path, target_path)

with open('preprocess.p', mode='rb') as in_file:
    source_int_text, target_int_text, source_vocab_to_int, target_vocab_to_int = pickle.load(in_file)

max_target_sentence_length = max([len(sentence) for sentence in source_int_text])
train_graph = tf.Graph()

with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()
    input_shape = tf.shape(input_data)
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, source_sequence_length, target_sequence_length, max_target_sequence_length, len(source_vocab_to_int), len(target_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
    
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        pad_targets_lengths = []
        
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))
            
        pad_source_lengths = []
        
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_accuracy(target, logits):
    max_seq = max(target.shape[1], logits.shape[1])
    
    if max_seq - target.shape[1]:
        target = np.pad(target, [(0,0),(0,max_seq - target.shape[1])], 'constant')
    
    if max_seq - logits.shape[1]:
        logits = np.pad(logits, [(0,0),(0,max_seq - logits.shape[1])], 'constant')
    
    return np.mean(np.equal(target, logits))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source, valid_target, batch_size, source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>']))

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch_i in range(epochs):
        
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(get_batches(train_source, train_target, batch_size, source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>'])):
            _, loss = sess.run([train_op, cost], {input_data: source_batch, targets: target_batch, lr: learning_rate, target_sequence_length: targets_lengths, source_sequence_length: sources_lengths, keep_prob: keep_probability})

            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(inference_logits, {input_data: source_batch, source_sequence_length: sources_lengths, target_sequence_length: targets_lengths, keep_prob: 1.0})
                batch_valid_logits = sess.run(inference_logits, {input_data: valid_sources_batch, source_sequence_length: valid_sources_lengths, target_sequence_length: valid_targets_lengths, keep_prob: 1.0})
                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)
                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
 
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
    
with open('params.p', 'wb') as out_file:
    pickle.dump(save_path, out_file)
    
with open('params.p', mode='rb') as in_file:
    load_path = pickle.load(in_file)
    
def sentence_to_seq(sentence, vocab_to_int):
    sentence = sentence.lower()
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split()]

translate_sentence = 'he saw a old yellow truck .'

translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size, target_sequence_length: [len(translate_sentence)*2]*batch_size, source_sequence_length: [len(translate_sentence)]*batch_size, keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))
print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))