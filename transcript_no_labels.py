import pandas as pd


#to return :

#dyad - session - begin_time - end_time - duration - P1 - P2 - cat colonnes
def do_the_job(path, save=False):
    """
    _______
    Input:
    Path to csv ending with '<path_to_folder>/D<dyad>_S<serie>.csv'
    _______
    Output:
    Cleant Dataframe if not save, None otherwise
    _______
    If save, the function will save the results at 'path_to_folder/cleant_D<dyad>_S<serie>.csv'

    We found out datasets sometimes show multiple times same lines when P1 and P2 speeches overlap.
    This function aims at solving this by grouping these rows into one row for P1 and one row for P2. 
    """

    #reading file
    raw=pd.read_csv(path)

    #selecting relevant columns
    df = raw[['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms', 'P1', 'P2']]

    #selecting relevant rows (i.e., )
    df=df.loc[ list(set(df.P2.dropna().index).union(set(df.P1.dropna().index))) ].reset_index(drop=True)

    #rows to datetime
    for i in range(2):
        df[df.columns[i]] = pd.to_datetime(df[df.columns[i]], format="%H:%M:%S.%f")

    #creating a columns that tracs whether lines repeat over axis 0
    df['P1d'] = df.P1.ne(df.P1.shift()).astype(int).cumsum()
    df['P2d'] = df.P2.ne(df.P2.shift()).astype(int).cumsum()

    #then groupby these new columns and the actual lines
    dfP1 = df.groupby(['P1', 'P1d']).agg(
        begin = ( df.columns[0] , 'first'),
        end   = ( df.columns[1] , 'last' )
    )
    dfP2 = df.groupby(['P2', 'P2d']).agg(
        begin = ( df.columns[0] , 'first'),
        end   = ( df.columns[1] , 'last' )
    )
    
    #calculate duration of each line ( could not find how to sum timedelta with pandas agg() )
    dfP1['dura'] = dfP1.end - dfP1.begin
    dfP2['dura'] = dfP2.end - dfP2.begin

    #get lines back to columns, drop created column, and sort rows by starting time
    dfP1 = dfP1.reset_index().drop(columns=['P1d']).set_index('begin')
    dfP2 = dfP2.reset_index().drop(columns=['P2d']).set_index('begin')

    #concatenate results and reset index
    results = pd.concat((dfP1, dfP2)).sort_index()
    results.reset_index(inplace=True, drop=False)

    #create two columns for external use
    results['Dyad'] = path.split('/')[-1].split('_')[0][1:]
    results['Session'] = path.split('/')[-1].split('_')[1].split('.')[0][1:]

    #reorder columns
    results = results.iloc[:, [5,6,0,2,3,1,4]]

    #reformat datetime columns back to original
    results.begin = results.begin.dt.strftime("%H:%M:%S.%f").apply(lambda x : x[:-3])
    results.end = results.end.dt.strftime("%H:%M:%S.%f").apply(lambda x : x[:-3])
    results.dura = results.dura.apply(lambda x:str(x)[7:-3])

    #rename columns
    results.columns= [
        'Dyad',
        'Session',
        raw.columns[0],
        raw.columns[1],
        raw.columns[2],
        'P1',
        'P2'
    ]

    #saving to csv?
    if save :
        results.to_csv( f"{ '/'.join( path.split('/')[:-1] ) }/cleant_{path.split('/')[-1]}")
        return None

    return results

def do_the_second_job(path, save=False):
    """
    _______
    Input:
    Path to csv ending with '<path_to_folder>/D<dyad>_S<serie>.csv'
    _______
    Output:
    Cleant Dataframe if not save, None otherwise
    _______
    If save, the function will save the results at 'path_to_folder/cleant_D<dyad>_S<serie>.csv'

    We found out datasets sometimes show multiple times same lines when P1 and P2 speeches overlap.
    This function aims at solving this by grouping these rows into one row for P1 and one row for P2. 
    """
    #reading file
    raw=pd.read_csv(path)

    #selecting relevant columns
    df = raw[['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms', 'Duration - hh:mm:ss.ms', 'P1', 'P2', 'SV1_P1', 'SV1_P2', 'SV2_P1', 'SV2_P2', 'Notes_SV']].copy()

    #selecting relevant rows (i.e., colunms where either P1 or P2 talks)
    df=df.loc[ list(
        set(
            df.P2.dropna().index
        ).union(set(
            df.P1.dropna().index
        )))]\
            .reset_index(drop=True)

    #rows to datetime
    for time_col in ['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms']:
        df[time_col] = pd.to_datetime(df[time_col], format="%H:%M:%S.%f")

    #creating a columns that tracs whether lines repeat over axis 0
    df['P1d'] = df.P1.ne(df.P1.shift()).astype(int).cumsum()
    df['P2d'] = df.P2.ne(df.P2.shift()).astype(int).cumsum()

    #then groupby these new columns and the actual lines
    dfP1 = df.groupby(['P1', 'P1d']).agg(
        begin    = ( 'Begin Time - hh:mm:ss.ms' , 'first'),
        end      = ( 'End Time - hh:mm:ss.ms'   , 'last' ),
        SV1_P1   = ( 'SV1_P1'   , 'first'),
        SV1_P2   = ( 'SV1_P2'   , 'first'),
        SV2_P1   = ( 'SV2_P1'   , 'first'),
        SV2_P2   = ( 'SV2_P2'   , 'first'),
        Notes_SV = ( 'Notes_SV' , 'first')
    )
    dfP2 = df.groupby(['P2', 'P2d']).agg(
        begin    = ( 'Begin Time - hh:mm:ss.ms' , 'first'),
        end      = ( 'End Time - hh:mm:ss.ms'   , 'last' ),
        SV1_P1   = ( 'SV1_P1'   , 'first'),
        SV1_P2   = ( 'SV1_P2'   , 'first'),
        SV2_P1   = ( 'SV2_P1'   , 'first'),
        SV2_P2   = ( 'SV2_P2'   , 'first'),
        Notes_SV = ( 'Notes_SV' , 'first')
    )

    #rounding ms to tenth of a second
    dfP1['begin'] = dfP1.begin.dt.round(freq='100L')
    dfP1['end'] = dfP1.end.dt.round(freq='100L')
    dfP2['begin'] = dfP2.begin.dt.round(freq='100L')
    dfP2['end'] = dfP2.end.dt.round(freq='100L')

    #calculate duration of each line ( could not find how to sum timedelta with pandas agg() )
    dfP1['dura'] = dfP1.end - dfP1.begin
    dfP2['dura'] = dfP2.end - dfP2.begin

    #get lines back to columns, drop created column, and sort rows by starting time
    dfP1 = dfP1.reset_index().drop(columns=['P1d']).set_index('begin')
    dfP2 = dfP2.reset_index().drop(columns=['P2d']).set_index('begin')

    #concatenate results and reset index
    results = pd.concat((dfP1, dfP2)).sort_index()
    results.reset_index(inplace=True, drop=False)

    #create two columns for external use
    results['Dyad'] = path.split("D")[-1].split("S")[0]
    results['Session'] = path.split("S")[-1].split('.')[0]

    #reformat datetime columns back to original
    results.begin = results.begin.dt.strftime("%H:%M:%S.%f").apply(lambda x : x[:-3])
    results.end = results.end.dt.strftime("%H:%M:%S.%f").apply(lambda x : x[:-3])
    results.dura = results.dura.apply(lambda x:str(x)[7:-3])

    #rename columns
    columns_dict= {
        'begin' : 'Begin Time - hh:mm:ss.ms',
        'end' : 'End Time - hh:mm:ss.ms',
        'dura' : 'Duration - hh:mm:ss.ms'
    }

    results.rename(columns=columns_dict, inplace=True)

    results = results[['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms', 'Duration - hh:mm:ss.ms', 'P1', 'P2', 'SV1_P1', 'SV1_P2', 'SV2_P1', 'SV2_P2', 'Notes_SV']]


    #saving to csv?
    if save :
        results.to_csv( f"{ '/'.join( path.split('/')[:-1] ) }/cleant_{path.split('/')[-1]}")
        return None

    return results

if __name__=="__main__":
    None