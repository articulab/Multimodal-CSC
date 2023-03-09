import pandas as pd
import datetime
import os
time_parser = lambda x : datetime.timedelta(seconds = float(x.replace(' ','')))
find_dyad = lambda path : path.split('/')[-1].split('_')[0].split('D')[-1]
find_session = lambda path : path.split('/')[-1].split('_')[1].split('S')[-1]
find_participant = lambda path : path.split('_')[-1][0]

def agg_open_face(openface_path="data/final/openface/") ->None:
    total={'A':[],'B':[]}
    openface_path="data/final/openface/"
    for path in [path for path in os.listdir(openface_path) if path[-3:]=='csv']:
        print(path)
        #retrieve dyad session and participant
        dyad = find_dyad(path)
        session = find_session(path)
        participant = find_participant(path)
        #retrive df and parse timestamps
        df = pd.read_csv(openface_path + path, parse_dates=[' timestamp'], date_parser=time_parser)
        #remove ' ' from col names
        df.columns = df.columns.str.replace(' ','')
        #add dyad and session column at the beginning
        columns = list(df.columns)
        df['Dyad'] = dyad
        df['Session'] = session
        df = df[['Dyad','Session'] + columns]
        #drop columns
        df = df.drop(columns=['frame','face_id','confidence','success'] + [col for col in df.columns if col [-2:]=='_c'] )
        #set dtypes
        df = df.astype({col:'float32' for col in df.columns if col [-2:]=='_r'})
        total[participant].append(df)
    total['A'] = pd.concat(total['A'], axis=0).reset_index(drop=True).to_feather(openface_path + "partA")
    total['B'] = pd.concat(total['B'], axis=0).reset_index(drop=True).to_feather(openface_path + "partB")

def agg_open_smile(opensmile_path="data/final/opensmile/") ->None:
    total={'A':[],'B':[]}
    for path in [path for path in os.listdir(opensmile_path) if path[-3:]=='csv']:
        print(path)
        #retrieve dyad session and participant
        dyad = find_dyad(path)
        session = find_session(path)
        participant = find_participant(path)
        #retrive df and parse timestamps
        df = pd.read_csv(opensmile_path + path, parse_dates=['frameTime'], date_parser=time_parser)
        #add dyad and session column at the beginning
        columns = list(df.columns)
        df['Dyad'] = dyad
        df['Session'] = session
        df = df[['Dyad','Session'] + columns]
        #drop columns
        df = df.drop(columns=['name'])
        #set dtypes
        df = df.astype({col:'float32' for col in df.columns if df[col].dtype=='float64'})
        total[participant].append(df)
    total['A'] = pd.concat(total['A'], axis=0).reset_index(drop=True).to_feather(opensmile_path + "partA")
    total['B'] = pd.concat(total['B'], axis=0).reset_index(drop=True).to_feather(opensmile_path + "partB")

def agg_on_05(df):
    df = df.set_index(['Dyad','Session','timestamp'])\
        .unstack(['Dyad','Session'])\
        .resample('500ms').mean()\
        .stack(['Session','Dyad'])\
        .swaplevel(0,2)\
        .sort_index(level=2)\
        .sort_index(level=1)\
        .sort_index(level=0)\
        .reset_index()
    return df