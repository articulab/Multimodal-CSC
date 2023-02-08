import pandas as pd
import os

def time_parser(date):
    try : date = pd.to_datetime(date, format='%H:%M:%S.%f')
    except : date = pd.to_datetime(date, format='%H:%M:%S')
    return date

def P1P2Merger(row):
    p1 = row.P1_x if type(row.P1_y)==float else row.P1_y
    p2 = row.P2_x if type(row.P2_y)==float else row.P2_y
    sim = not((row.P1_x == row.P1_y) or (row.P2_y == row.P2_x))
    return pd.Series([p1,p2,sim])

def merger(path_full, path_vsn, dyad, session):

    print('doing ', path_vsn)

    full = pd.read_csv(path_full)
    full['Time_begin'] = full['Time_begin'].apply(time_parser).dt.round(freq='100L')
    full['Time_end'] = full['Time_end'].apply(time_parser).dt.round(freq='100L')

    print(dyad, session)
    print(full["SD_Tutor"].unique())
    print(full["SD_Tutee"].unique())
    print(full["PR_Tutor"].unique())
    print(full["PR_Tutee"].unique())

    samp = full.loc[(full.Dyad==dyad) & (full.Session==session)]

    print(dyad, session)
    print(len(samp))
    print(samp["SD_Tutor"].unique())
    print(samp["SD_Tutee"].unique())
    print(samp["PR_Tutor"].unique())
    print(samp["PR_Tutee"].unique())

    vsn = pd.read_csv(path_vsn)
    vsn['Begin Time - hh:mm:ss.ms'] = pd.to_datetime(vsn['Begin Time - hh:mm:ss.ms'], format='%H:%M:%S.%f').dt.round(freq='100L')
    vsn['End Time - hh:mm:ss.ms'] = pd.to_datetime(vsn['End Time - hh:mm:ss.ms'], format='%H:%M:%S.%f').dt.round(freq='100L')
    vsn['Duration - hh:mm:ss.ms'] = pd.to_datetime(vsn['Duration - hh:mm:ss.ms'], format='%H:%M:%S.%f').dt.round(freq='100L')

    vsn = vsn[['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms',
        'Duration - hh:mm:ss.ms', 'PP_Notes', 'SV1_P1', 'SV1_P2',
        'SV2_P1', 'SV2_P2', 'PP_P2',
        'P1', 'P2', 'PP_P1']]

    try : vsn.columns = ['Begin_time', 'End_time',
        'Duration', 'PP_Notes', 'SV1_P1', 'SV1_P2',
        'SV2_P1', 'SV2_P2', 'PP_P2', 'P1', 'P2','PP_P1']
    
    except Exception as e:
        print(e)
        return pd.DataFrame()


    rez = pd.merge(
        samp,
        vsn,
        left_on='Time_begin',
        right_on='Begin_time',
        how='outer'
    )

    print(rez["SD_Tutor"].unique())

    rez['btime'] = rez.apply(lambda row: row.Begin_time if type(row.Time_begin)==pd._libs.tslibs.nattype.NaTType else row.Time_begin, axis=1)
    rez['etime'] = rez.apply(lambda row: row.End_time if type(row.Time_end)==pd._libs.tslibs.nattype.NaTType else row.Time_end, axis=1)
    rez.sort_values(by='btime',inplace=True)
    rez[['P1', 'P2','sim']] = rez.apply(P1P2Merger, axis=1)
    rez.drop(columns=['Time_begin', 'Begin_time','Time_end', 'End_time', 'P1_x', 'P2_x', 'P1_y', 'P2_y'], inplace=True)

    rez['P1d'] = rez.P1.ne(rez.P1.shift()).astype(int).cumsum()
    rez['P2d'] = rez.P2.ne(rez.P2.shift()).astype(int).cumsum()

    def customMerger(s):
        try : return s.dropna().iloc[0]
        except : return pd.NA

    full1 = rez.groupby(['P1', 'P1d']).agg(
        Dyad            = ('Dyad', customMerger),
        Session         = ('Session', customMerger),
        Period          = ('Period', customMerger),
        Begin_time      = ('btime','first'),
        End_time        = ('etime','last'),
        Duration        = ('Duration', customMerger),
        SV1_P1          = ('SV1_P1', customMerger),
        SV1_P2          = ('SV1_P2', customMerger),
        SV2_P1          = ('SV2_P1', customMerger),
        SV2_P2          = ('SV2_P2', customMerger),
        HD_Tutee        = ('HD_Tutee', customMerger),
        HD_Tutor        = ('HD_Tutor', customMerger),
        PR_Tutee        = ('PR_Tutee', customMerger),
        PR_Tutor        = ('PR_Tutor', customMerger),
        SD_Tutee        = ('SD_Tutee', customMerger),
        SD_Tutor        = ('SD_Tutor', customMerger),
    ).reset_index(drop=False)

    full2 = rez.groupby(['P2', 'P2d']).agg(
        Dyad            = ('Dyad', customMerger),
        Session         = ('Session', customMerger),
        Period          = ('Period', customMerger),
        Begin_time      = ('btime','first'),
        End_time        = ('etime','last'),
        Duration        = ('Duration', customMerger),
        SV1_P1          = ('SV1_P1', customMerger),
        SV1_P2          = ('SV1_P2', customMerger),
        SV2_P1          = ('SV2_P1', customMerger),
        SV2_P2          = ('SV2_P2', customMerger),
        HD_Tutee        = ('HD_Tutee', customMerger),
        HD_Tutor        = ('HD_Tutor', customMerger),
        PR_Tutee        = ('PR_Tutee', customMerger),
        PR_Tutor        = ('PR_Tutor', customMerger),
        SD_Tutee        = ('SD_Tutee', customMerger),
        SD_Tutor        = ('SD_Tutor', customMerger),
    ).reset_index(drop=False)

    fin = pd.concat((full1,full2), axis=0).sort_values(by='Begin_time').reset_index(drop=True).drop(columns=['P1d','P2d'])
    fin = fin[['Dyad', 'Session', 'Period', 'Begin_time', 'End_time', 'Duration',
        'P1', 'P2', 'SV2_P1', 'SV2_P2', 'HD_Tutee', 'HD_Tutor',
        'PR_Tutee', 'PR_Tutor', 'SD_Tutee', 'SD_Tutor']]

    fin["Dyad"] = dyad
    fin["Session"] = session
    fin["Period"] = fin["Period"].replace(pd.NA, "").apply(lambda x : x.strip())
    fin['Begin_time'] = fin.Begin_time.dt.strftime('%H:%M:%S.%f')
    fin['End_time'] = fin.End_time.dt.strftime('%H:%M:%S.%f')
    fin['Duration'] = pd.to_datetime(fin.Duration)
    fin['Duration'] = fin.Duration.dt.strftime('%H:%M:%S.%f')

    print('did ', path_vsn)
    
    return fin

def find_DS(vsn):
    return int(vsn.split('_')[-1].split('D')[-1].split('S')[0]), int(vsn.split('_')[-1].split('S')[-1].split('.')[0])

def vsn_files_and_DS(dir):
    return [(f"{dir}/{vsn}",find_DS(vsn)) for vsn in os.listdir(dir) if vsn[-3:]=='csv'] 

def dir_merger(full_behav, dir):

    vsn_and_DS = vsn_files_and_DS(dir)

    result = [merger(full_behav, vsn, dyad, session) for (vsn, (dyad, session)) in vsn_and_DS]

    return pd.concat(result, axis=0)


if __name__ == "__main__":
    full_df = "data/transcripts/2016_full_behaviors_annotation_w_hedges.csv"
    sv_dir = "data/transcripts/sv_detail/"

    final_df = dir_merger(full_df, sv_dir)

    final_df.rename(columns = {"SV2_P1" : "SV_Tutor", "SV2_P2" : "SV_Tutee"})
    final_df.to_csv("merged_df_2016.csv", index = False)