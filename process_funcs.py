import os
import numpy as np
import pandas as pd
import pickle
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

"""GENERAL FUNCS======================================================"""
#Get train/test data 
def get_train_test():
    names = ['train_x', 'test_x', 'train_y', 'test_y', 'label_map']
    variables = []
    for n in names: 
        with open(f'vars/{n}.pkl', 'rb') as f: 
            variables.append(pickle.load(f))
    return variables

"""EMBEDDING-SPECIFIC FUNCS======================================================"""
#Generate doc vectors using word2vec/fasttext
def create_doc_vecs(docs, word_vecs): 
    try:
        word_vec_dim = word_vecs['test'].shape
    except:
        word_vec_dim = word_vecs[0].shape

    doc_vecs = []
    for doc in docs: 
        n_words = 0
        doc_vec = np.zeros(word_vec_dim)
        for word in doc:
            try:
                doc_vec = np.add(doc_vec, word_vecs[word])
                n_words += 1
            except: #Word not found in word2vec vocab
                continue
        doc_vec = doc_vec / n_words
        doc_vecs.append(doc_vec)
        
    return doc_vecs

#GLOVE
def get_glove_embeddings(): 
    glove_dir = 'embeddings\glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding = 'utf8')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    return embeddings_index

#WORD2VEC
def get_word2vec_embeddings():
    word2vec_dir = 'embeddings\GoogleNews-vectors-negative300.bin.gz'
    word_vecs = gensim.models.KeyedVectors.load_word2vec_format(word2vec_dir, binary = True)
    embeddings_index = {}
    for word, idx in word_vecs.key_to_index.items():
        embeddings_index[word] = word_vecs[word]

    return embeddings_index    

#FASTTEXT
def get_fasttext_embeddings():
    fasttext_dir = 'embeddings\wiki-news-300d-1M.vec'
    embeddings_index = {}
    with open(fasttext_dir, encoding = 'utf8', newline='\n', errors='ignore') as f:
        for line in f: 
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    return embeddings_index

#CUSTOM EMBEDDINGS
def get_custom_embeddings(): 
    word2vec_dir = 'embeddings\custom_embeddings.model'
    word_vecs = gensim.models.Word2Vec.load(word2vec_dir)
    embeddings_index = {}
    for word in word_vecs.wv.key_to_index:
        embeddings_index[word] = word_vecs.wv[word]

    return embeddings_index    

"""MODEL-SPECIFIC FUNCS======================================================"""
def lstm_build_sequences(data, max_words, max_len, tokenizer = None): 
    if tokenizer == None: #fit new tokenizer
        tokenizer = Tokenizer(num_words=max_words, lower=False)
        tokenizer.fit_on_texts(data)

    sequences = tokenizer.texts_to_sequences(data)
    sequences = pad_sequences(sequences, maxlen = max_len)
    return sequences, tokenizer

def lstm_build_embed_mat(embeddings_index, word_index, max_words, embedding_dim = None):
    if embedding_dim == None: 
        embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]]) #sampling 1st item of word vectors
    embedding_mat = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words: 
            embedding_vec = embeddings_index.get(word)
            if embedding_vec is not None: 
                embedding_mat[i] = embedding_vec
    return embedding_dim, embedding_mat

def lstm_plot(model, history, test_x, test_y):
    acc = history.history['acc'][:-1]
    val_acc = history.history['val_acc'][:-1]
    loss = history.history['loss'][:-1]
    val_loss = history.history['val_loss'][:-1]
    epochs = range(0, len(acc))

    #Accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    #Loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    #TESTING instances
    y_pred_arrs = model.predict(test_x)
    y_preds = []
    for pred in y_pred_arrs: 
        y_preds.append(np.argmax(pred))
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    weighted_f1 = np.round(f1_score(test_y, y_preds, average = 'weighted'), 4)

    return accuracy, weighted_f1

def run_model(model, train_x, train_y, test_x, test_y, label_map): 
    #Model run - for SVM/XGB models
    model.fit(train_x, train_y)
    y_preds = model.predict(test_x)
    
    #Reporting
    target_names = [k for k,v in label_map.items()]
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    weighted_f1 = np.round(f1_score(test_y, y_preds, average = 'weighted'), 4)
    print("ACCURACY %: ", accuracy)
    print("WEIGHTED F1: ", weighted_f1)
    print(classification_report(test_y, y_preds, target_names = target_names))

    return model, y_preds, accuracy, weighted_f1


def store_results(model_name, emb_name, model, accuracy, f1_score, tf_model = 0): 
    #Save model
    model_concat = f'{model_name}-{emb_name}'
    if tf_model: 
        model.save(f'models/{model_concat}')
    else:
        with open(f'models/{model_concat}.pkl', 'wb') as f: 
            pickle.dump(model, f)

    #Record results
    file_name = 'results.csv'
    results_df = pd.read_csv(file_name)
    if model_concat in results_df['model'].values: 
        drop_mask = results_df.model == model_concat
        results_df.drop(results_df[drop_mask].index, inplace=True)
    results_df = results_df.append({
                            'model': model_concat, 
                            'accuracy': accuracy, 
                            'weighted_f1': f1_score}, 
                            ignore_index=True)
    # results_df.reset_index(inplace=True)
    results_df.to_csv(file_name, index=False)