import pandas as pd

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
    df = raw[[raw.columns[0], raw.columns[1], 'P1', 'P2']]

    #selecting relevant rows
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
    results.dura = results.dura.apply(lambda x: int(len(str(x)) == len('0 days 00:00:01')) * (str(x)[7:] + '.000') + int(len(str(x)) != len('0 days 00:00:01')) * str(x)[7: -3])

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
        results.to_csv("clean_" + file)
        return None

    return results

if __name__ == "__main__":
    import os
    for file in os.listdir():
        if file.endswith(".csv") and file.startswith("D"):
            do_the_job(file, save = True)