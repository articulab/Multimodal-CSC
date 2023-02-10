import numpy as np
import pandas as pd
import torch
import transformers
from train_text import read_dataset

def create_utt_df( df:pd.DataFrame, cats_col = ['VSN', 'PR', 'SD', 'QE', 'HD','None'] ) ->pd.DataFrame :
    """
    Takes df with P1, P2, *cats columns and returns pd.df[['utt':utterances, 'speaker':(1=P1 or 2=P2), [cats...]]]
    """
    merge = lambda tp : [tp[0],1] if tp[1]=='_' else [tp[1],2]
    utts = pd.DataFrame(
        [ *map(
            merge,
            zip( df.P1.fillna('_'), df.P2.fillna('_') )
        ) ],
        columns=['utt','speaker']
    )
    utt_df = pd.concat(
        (
            utts.astype({'speaker':'category'}),
            df[cats_col].astype('category')
        ),
        axis=1
    )
    return utt_df

def embed_utts( utts:pd.DataFrame, model_name="distilbert-base-uncased", tokenizer_name="distilbert-base-uncased", MAX_LEN=128) ->pd.DataFrame :
    """Takes utterances_df and return dataframe of embeddings"""

    model     = transformers.DistilBertModel.from_pretrained(model_name)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained( tokenizer_name )

    #tokenize one utt
    tokenize = lambda text : tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
        )
    #takes output of tokenize and calculate output of model
    embed = lambda tk : model(input_ids = tk[0], attention_mask = tk[1])[0][:, 0].detach().numpy().flatten()

    #use tokenize and embed functions to create embedings
    embeds = pd.DataFrame(
        [ embed( (tk['input_ids'], tk['attention_mask']) ) for tk in map(tokenize, utts.utt) ]
    )
    embeds = embeds.rename(columns={col:str(col) for col in embeds.columns}) #we want str col names

    final= pd.concat((utts,embeds), axis=1).reset_index()

    return final

def pipeline(
    text_path = "data/merged_df_2016.csv",
    cats_col = ['VSN', 'PR', 'SD', 'QE', 'HD','None'],
    output_path = "data/merged_df_2016_embeds",
    model_name = "distilbert-base-uncased",
    tokenizer_name = "distilbert-base-uncased",
    MAX_LEN = 128 #max len is also the model's input size
):
    """Make embedings from dataset and create embedings
    ==Inputs==
    text_path:str path to dataset
    cats_col:list of cats columns un dataset
    output_path:str path to output
    model_name:str will be given to DistilBertModel.from_pretrained
    model_name:str will be given to DistilBertTokenizer.from_pretrained
    MAX_LEN that is also the model's input size
    ==Outputs==
    pd.DataFrame: ( [ 'index', 'utt':utterances, 'speaker':(1=P1 or 2=P2), [*cats_col], [*range(MAX_LEN)] ] )
    """

    #import dataset
    df = pd.read_csv( text_path )

    #transforms it like read_dataset from train_text.py
    df = read_dataset( text_path )

    #start final df:
    utt_df = create_utt_df( df, cats_col )

    #add embedings to utt_df
    final = embed_utts( utt_df, model_name=model_name, tokenizer_name=tokenizer_name, MAX_LEN=MAX_LEN )

    final.to_feather(output_path)

    return final

if __name__=='__main__':
    pipeline() #time approx: 14 s ± 752 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for 100