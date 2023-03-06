import pandas as pd
import numpy as np
from train_text import read_dataset
import transformers
from tqdm import tqdm

def get_annotated_dataset(dir = "final_data_handmatched_2016.csv"):
    return pd.read_csv(dir).reset_index()


def create_utt_df( df:pd.DataFrame, cats_col = ['SD','QE','SV','PR','HD'] ) -> pd.DataFrame :
    """
    Takes df with P1, P2, *cats columns and returns pd.df[['utt':utterances, 'speaker':(1=P1 or 2=P2), [cats...]]]
    """
    merge = lambda tp : [tp[0],1] if tp[1] == '_' else [tp[1],2]
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


def embed_utts( utts:pd.DataFrame, model_name="distilbert-base-uncased", tokenizer_name="distilbert-base-uncased", MAX_LEN=128) :
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
    text_path = "final_data_handmatched_2016.csv",
    cats_col = ['SD','QE','SV','PR','HD'],
    output_path = "bert_embeddings",
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
    df = pd.read_csv(text_path).reset_index()
    df["None"] = pd.Series()
    df.loc[(df['SD'].isnull() & df['QE'].isnull() & df['SV'].isnull() & df['PR'].isnull() & df['HD'].isnull()), "None"] = "x"
    #transforms it like read_dataset from train_text.py
    # df = read_dataset(text_path)

    #start final df:
    utt_df = create_utt_df(df, cats_col)

    #add embedings to utt_df
    final = embed_utts(utt_df, model_name=model_name, tokenizer_name=tokenizer_name, MAX_LEN=MAX_LEN )

    final.to_feather(output_path)
    final.to_csv(output_path + ".csv")

    return final


def myround(x, prec = 1, base = .5):
    """
    Rounds floats

    Args:
        x (_type_): float
        prec (int, optional): number of decimal places. Defaults to 1.
        base (float, optional): value to round to. Defaults to .5.

    Returns:
        _type_: _description_
    """
    return round(base * round(float(x)/base),prec)


def get_opensmile_data(dyad, session, p, dir = "data/audio/opensmile", precision = .5, method = "median"):
    """
    Given a dyad, session and participant, returns the mean/median values of the dataframe corresponding to this person audio features.

    Args:
        dyad (_type_): dyad number
        session (_type_): session number
        p (_type_): 1 or 2 given the speaker
        dir (str, optional): Path to the folder containing opensmile tables. Defaults to "data/audio/opensmile".
        precision (float, optional): Time slices on which aggregating feature values. Defaults to .5.
        method (str, optional): Method of aggregating. Defaults to "median".
    """

    participant = "A" * int(int(p) == 1) + "B" * int(int(p) == 2)
    opensmile_df = pd.read_csv(dir + f"/D{dyad}_S{session}_Participant_{participant}.csv")

    opensmile_df["frameTime"] = opensmile_df["frameTime"].apply(lambda x : myround(x, 1, precision))

    if method == "median":
        opensmile_df = opensmile_df.groupby("frameTime").median().reset_index()
    elif method == "mean":
        opensmile_df = opensmile_df.groupby("frameTime").mean().reset_index()

    return opensmile_df


def get_openface_data(dyad, session, p, dir = "data/video/openface", precision = .5, method = "median"):
    """
    Given a dyad, session and participant, returns the mean/median values of the dataframe corresponding to this person video features.

    Args:
        dyad (_type_): dyad number
        session (_type_): session number
        p (_type_): 1 or 2 given the speaker
        dir (str, optional): Path to the folder containing opensmile tables. Defaults to "data/video/openface".
        precision (float, optional): Time slices on which aggregating feature values. Defaults to .5.
        method (str, optional): Method of aggregating. Defaults to "median".
    """

    participant = "A" * int(int(p) == 1) + "B" * int(int(p) == 2)
    openface_df = pd.read_csv(dir + f"/D{dyad}_S{session}_Participant_{participant}.csv")

    openface_df[" timestamp"] = openface_df[" timestamp"].apply(lambda x : myround(x, 1, precision))

    if method == "median":
        openface_df = openface_df.groupby(" timestamp").median().reset_index()
    elif method == "mean":
        openface_df = openface_df.groupby(" timestamp").mean().reset_index()

    return openface_df


def hhmmssms_to_s(ts):
    """
    Converts hh:mm:ss.ms format to second float count.

    Args:
        ts (_type_): hh:mm:ss.ms format

    Returns:
        s_count: rounded second values
    """
    h, m, s = ts.split(".")[0].split(":")
    ms = ts.split(".")[1]

    s_count = int(ms) / 1000 + int(s) + 60 * (int(m) + 60 * int(h))
    return myround(s_count)

def merge_modalities(main_df_dir = "FUZZY_MATCH_CORRECTED_2016.csv", audio_time_window = 12, video_time_window = 12):
    """
    Given the main dataframe containing "Dyad", "Session", "P1", "P2", "Begin_time", "End_time", and categories to label, will add the openface and opensmile features to each line (heavy)

    Args:
        main_df_dir (str, optional): Path to the csv. Defaults to "final_data_handmatched_2016.csv".
        audio_time_window (int, optional): Time frame to consider for audio feature retrieval. Defaults to 10.
        video_time_window (int, optional): Time frame to consider for video feature retrieval. Defaults to 10.
    """

    df = get_annotated_dataset().sort_values(by = ["Dyad", "Session", "Begin_time"])
    audio_df = get_opensmile_data(3, 2, 1)
    video_df = get_openface_data(3, 2, 1)

    # Here we will create a separate column for each value at each time frame.
    audio_columns = list(audio_df.columns)[2:]
    video_columns = list(video_df.columns)[5:22]

    audio_features = []
    video_features = []

    for i in range(2 * audio_time_window):
        audio_features += [c + str(i) for c in audio_columns]
    for i in range(2 * video_time_window):
        video_features += [c + str(i) for c in video_columns]

    df["Second_count"] = df["End_time"].apply(hhmmssms_to_s)

    df[audio_features] = [pd.NA for _ in audio_features]
    df[video_features] = [pd.NA for _ in video_features]
    dyads_sessions = [(3,1), (3,2), (4,1), (4,2), (5, 1), (5,2), (6,1), (6,2), (7,1), (7,2), (8,2), (10,1), (10,2)]
    problem_count = 1

    for (dyad, session) in tqdm(dyads_sessions):
        mask = (df["Dyad"] == dyad) & (df["Session"] == session) & (df["Second_count"] > audio_time_window)
        current_df = df[mask]

        audio_df_1 = get_opensmile_data(dyad, session, 1)
        audio_df_2 = get_opensmile_data(dyad, session, 2)

        video_df_1 = get_openface_data(dyad, session, 1)
        video_df_2 = get_openface_data(dyad, session, 2)

        audio_df_dict = {1 : audio_df_1, 2 : audio_df_2}
        video_df_dict = {1 : video_df_1, 2 : video_df_2}

        for idx, row in tqdm(current_df.iterrows()):
            e_time = row["Second_count"]
            p = 2
            if pd.isna(row["P2"]):
                p = 1
            audio_sel_df = audio_df_dict[p]
            video_sel_df = video_df_dict[p]

            audio_mask_time = (audio_sel_df["frameTime"] <= e_time) & (audio_sel_df["frameTime"] > e_time - 12)
            opensmile_features = audio_sel_df.loc[audio_mask_time, audio_columns].values

            video_mask_time = (video_sel_df[" timestamp"] <= e_time) & (video_sel_df[" timestamp"] > e_time - 12)
            openface_features = video_sel_df.loc[video_mask_time, video_columns].values
            try:
                current_df.loc[idx, audio_features] = opensmile_features.flatten()
            except:
                current_df.loc[idx, audio_features] = 0
                print(f"==============\nPROBLEM NUMBER {problem_count} AT ROW==============\n")
                print(row)
                problem_count += 1

            try:
                current_df.loc[idx, video_features] = openface_features.flatten()
            except:
                current_df.loc[idx, video_features] = 0
                print(f"==============\nPROBLEM NUMBER {problem_count} AT ROW==============\n")
                print(row)
                problem_count += 1
        df.loc[mask] = current_df
    df.to_csv("audio_video_features_test.csv")
            


if __name__ == "__main__":
    # pipeline()
    merge_modalities()