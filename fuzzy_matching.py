from fuzzywuzzy import fuzz, process
import pandas as pd
import numpy as np
import os
import tqdm 

def get_corrected_df(dir = "./data/transcripts/"):
    """
    Script that takes all the corrected dataframes and merges them in a big one
    """
    cols = pd.read_csv("data/transcripts/clean_D3_S1.csv").columns
    df_time = pd.DataFrame(columns = cols)

    for file in os.listdir(dir):
        if file.startswith("clean_") and file.endswith('.csv'):
            curr_df = pd.read_csv(dir + file)
            if "Temps de départ - hh:mm:ss.ms" in curr_df.columns:
                curr_df["Begin Time - hh:mm:ss.ms"] = curr_df["Temps de départ - hh:mm:ss.ms"]
                curr_df = curr_df.drop(columns = ["Temps de départ - hh:mm:ss.ms"])
            if 'temps de fin - hh:mm:ss.ms' in curr_df.columns:
                curr_df['End Time - hh:mm:ss.ms'] = curr_df['temps de fin - hh:mm:ss.ms']
                curr_df = curr_df.drop(columns = ['temps de fin - hh:mm:ss.ms'])
            if 'Durée - hh:mm:ss.ms' in curr_df.columns:
                curr_df['Duration - hh:mm:ss.ms'] = curr_df['Durée - hh:mm:ss.ms']
                curr_df = curr_df.drop(columns = ['Durée - hh:mm:ss.ms'])
            if ("jkakaka" in file and "1" in file) or ("8" in file and "2" in file):
                curr_df.loc[:, ["P1", "P2"]] = curr_df.loc[:, ["P2", "P1"]].values
                print("FOUND BUG") 
                print(file)
            df_time = pd.concat([df_time, curr_df])

    df_time['Time_begin_ts'] = pd.to_datetime(df_time["Begin Time - hh:mm:ss.ms"], format = "%H:%M:%S.%f")
    df_time = df_time.drop(columns = ["Unnamed: 0"])
    df_time = df_time[['Dyad', 'Session', 'Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms', 'Duration - hh:mm:ss.ms', 'P1', 'P2']]
    df_time['Time_begin_ts'] = pd.to_datetime(df_time["Begin Time - hh:mm:ss.ms"], format = "%H:%M:%S.%f")
    mask = ((df_time.Dyad == 6) & (df_time.Session == 1))
    # print(df_time.info())
    # print(type(df_time.iloc[0]["Dyad"]), type(df_time.iloc[0]["Session"]))
    # df_source.loc[mask, ["P1", "P2"]] = pd.concat([df_source.loc[mask].P2, df_source.loc[mask].P1], axis = 1)
    df_time.loc[mask, ["P1", "P2"]] = df_time.loc[mask, ["P2", "P1"]].values
    
    return df_time

def get_annotations_df(dir = "merged_df_2016.csv"):
    df_source = pd.read_csv(dir)

    # Get timestamps for beginning time
    df_source["Time_begin_ts"] = pd.to_datetime(df_source["Begin_time"], format = "%H:%M:%S.%f")
    df_source = df_source.sort_values(by = ["Dyad", "Session", "Begin_time"])

    # Select the annotations to keep - omitting SV1A which is off-task talk
    sd_df = df_source[(df_source["SD_Tutor"] == "SD") | (df_source["SD_Tutee"] == "SD")].replace("SD", "x").fillna("")
    qe_df = df_source[(df_source["SD_Tutor"] == "QE") | (df_source["SD_Tutee"] == "QE")].replace("QE", "x").fillna("")
    sv_df = df_source[~((df_source["SV2_P1"].isin(["SV1A", "", pd.NA])) | (df_source["SV2_P2"].isin(["SV1A", "", pd.NA])))].replace(['SV1C', 'SV2A', 'SV3A', 'SV1D', 'SV2B', 'SV3C', 'SV1B','SV2C'], "x").fillna("")
    pr_df = df_source[(df_source["PR_Tutor"].isin(['LPA', 'UL', 'LPP', 'LPB', '0', 'LP'])) | (df_source["PR_Tutee"].isin(['LPA', 'UL', 'LPP', 'LPB', '0', 'LP']))].replace(['LPA', 'UL', 'LPP', 'LPB', '0', 'LP'], "x").fillna("")
    hd_df = df_source[(df_source["HD_Tutor"].isin(['IDQ/IDS', 'IDQ', 'IDS', 'IDE', 'IDA', 'IDA/IDQ', 'IDQ/IDE', 'IDE/IDS'])) | (df_source["HD_Tutee"].isin(['IDQ/IDS', 'IDQ', 'IDS', 'IDE', 'IDA', 'IDA/IDQ', 'IDQ/IDE', 'IDE/IDS']))].replace(['IDQ/IDS', 'IDQ', 'IDS', 'IDE', 'IDA', 'IDA/IDQ', 'IDQ/IDE', 'IDE/IDS'], "x").fillna("")

    # No need to keep the information of who the strategy concerns, as it is the one that speaks at the time
    sd_df["SD"] = sd_df.SD_Tutor + sd_df.SD_Tutee
    qe_df["QE"] = qe_df.SD_Tutor + qe_df.SD_Tutee
    sv_df["SV"] = sv_df.SV2_P1 + sv_df.SV2_P2
    pr_df["PR"] = pr_df.PR_Tutor + pr_df.PR_Tutee
    hd_df["HD"] = hd_df.HD_Tutor + hd_df.HD_Tutee

    sd_map_dict = dict(zip(sd_df.index, sd_df["SD"]))
    qe_map_dict = dict(zip(qe_df.index, qe_df["QE"]))
    sv_map_dict = dict(zip(sv_df.index, sv_df["SV"]))
    pr_map_dict = dict(zip(pr_df.index, pr_df["PR"]))
    hd_map_dict = dict(zip(hd_df.index, hd_df["HD"]))

    df_source["SD"] = df_source.index.to_series().map(sd_map_dict)
    df_source["QE"] = df_source.index.to_series().map(qe_map_dict)
    df_source["SV"] = df_source.index.to_series().map(sv_map_dict)
    df_source["PR"] = df_source.index.to_series().map(pr_map_dict)
    df_source["HD"] = df_source.index.to_series().map(hd_map_dict)

    df_source = df_source.drop(df_source[df_source["SV"] == "xx"].index)
    
    return df_source

def fuzzy_match(df_annotations, df_corrected):
    
    df_source = df_annotations
    df_time = df_corrected

    result = []
    dyad_list = [3, 4, 5, 6, 7, 8, 10]
    session_list = [1, 2]
    threshold = 20
    
    df_source.loc[:, ["P1", "P2"]].replace("", pd.NA, inplace = True)
    df_source = df_source[(df_source["P1"].notna()) ^ (df_source["P2"].notna())]

    df_source["Time_begin_ts"] = pd.to_datetime(df_source["Begin_time"], format = "%H:%M:%S.%f")
    df_time['Time_begin_ts'] = pd.to_datetime(df_time["Begin Time - hh:mm:ss.ms"], format = "%H:%M:%S.%f")

    none_err_count = 0
    cl_err_count = 0
    for dyad_ in dyad_list:
        for session_ in session_list:
            # Selecting the subsets of the dataframe corresponding to current dyad/session pairs
            i_sel_s = (df_source["Dyad"] == dyad_) & (df_source["Session"] == session_)
            i_sel_s &= (df_source[f"P1"].notna()) | (df_source[f"P2"].notna())

            i_sel_t = (df_time["Dyad"] == dyad_) & (df_time["Session"] == session_)
            i_sel_t &= (df_time[f"P1"].notna()) | (df_time[f"P2"].notna())
            curr_df_source_ = df_source[i_sel_s]
            curr_df_time_ = df_time[i_sel_t]

            # Adding identifiers at the start of the strings - convenient do not mind this part
            curr_df_source_.loc[curr_df_source_["P1"].notna(), "P1"] = "1" + curr_df_source_["P1"]
            curr_df_source_.loc[curr_df_source_["P2"].notna(), "P2"] = "2" + curr_df_source_["P2"]

            curr_df_source_ = curr_df_source_.fillna('')
            curr_df_source_["P"] = curr_df_source_["P1"] + curr_df_source_["P2"]

            source_P = curr_df_source_["P"].tolist()
            source_timestamps_begin, source_timestamps_end = curr_df_source_["Begin_time"], curr_df_source_["End_time"] # mm:ss.ms
            source_classes = curr_df_source_[["SD", "QE", "SV", "PR", "HD"]].values.tolist()

            for utt_, bts_, ets_, cl_ in zip(source_P, source_timestamps_begin, source_timestamps_end, source_classes):
                bts_ = pd.to_datetime(bts_, format = "%H:%M:%S.%f")
                # utt_ = utt_.replace("pause filler", "").replace("(laughter)", "")
                cl_ = [len(c_) * cs_ for (c_, cs_) in zip (cl_, ["SD", "QE", "SV", "PR", "HD"])]

                curr_df_time_fz_ = curr_df_time_[(curr_df_time_["Time_begin_ts"] < bts_ + pd.Timedelta(seconds = 60)) & (curr_df_time_["Time_begin_ts"] > bts_ - pd.Timedelta(seconds = 60))]
                time_P1 = curr_df_time_fz_["P1"].apply(lambda x : int(str(x) != "nan") * (str(x).strip("\""))).tolist()
                time_P2 = curr_df_time_fz_["P2"].apply(lambda x : int(str(x) != "nan") * (str(x).strip("\""))).tolist()

                if utt_[0] == "1":
                    match = process.extract(utt_[1:], time_P1, limit = 3)
                    indexes = get_index(match, curr_df_time_fz_, "1")
                    try:
                        idx = list(list(curr_df_time_fz_[curr_df_time_fz_["P1"] == match[0][0]][["Dyad", "Session", 'Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms']].values)[0])
                        result.append({"person": 1, "match" : match, "informations" : idx, "index" : list(indexes)})
                    except : 
                        result.append({"person": 1, "match" : "", "informations" : "", "index" : 0})
                        if cl_ == "":
                            cl_err_count += 1
                        else : none_err_count += 1
                    
                elif utt_[0] == "2":
                    match = process.extract(utt_[1:], time_P2, limit = 3)
                    indexes = get_index(match, curr_df_time_fz_, "2")
                    try:
                        idx = list(list(curr_df_time_fz_[curr_df_time_fz_["P2"] == match[0][0]][["Dyad", "Session", 'Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms']].values)[0])
                        result.append({"person": 2, "match" : match, "informations" : idx, "index" : list(indexes)})
                    except : 
                        result.append({"person": 2, "match" : "", "informations" : "", "index" : 0})
                        if cl_ == "":
                            cl_err_count += 1
                        else : none_err_count += 1

    df_source_result = df_source[((df_source["P1"].notna()) ^ (df_source["P2"].notna())) & (df_source["Dyad"].isin(dyad_list)) & (df_source["Session"].isin(session_list))]
    df_source_result["Match"] = result
    print(f"None Errors  : {none_err_count}, Class Errors  : {cl_err_count}")
    return df_source_result

def max_match(series):
    match = series["match"]
    try:
        return max(match[0][1], match[1][1], match[2][1])
    except:
        try:
            return max(match[0][1], match[1][1])
        except:
            try:
                return match[0][1]
            except:
                return 0

def get_index(match, df, person):
    try:
        idx = df.index[df["P" + person] == match[0][0]].tolist() + df.index[df["P" + person] == match[1][0]].tolist() + df.index[df["P" + person] == match[2][0]].tolist()
        return idx
    except:
        try:
            df.index[df["P" + person] == match[0][0]].tolist() + df.index[df["P" + person] == match[1][0]].tolist()
            return idx
        except:
            try:
                df.index[df["P" + person] == match[0][0]].tolist()
                return idx
            except:
                idx = None
                return idx

def get_final_df(dir = "./data/transcripts/"):
    """
    Script that takes all the corrected dataframes and merges them in a big one
    """
    cols = pd.read_csv("data/transcripts/clean_D3_S1.csv").columns
    df_time = pd.DataFrame(columns = cols)

    for file in os.listdir(dir):
        if file.startswith("clean_") and file.endswith('.csv'):
            curr_df = pd.read_csv(dir + file)
            if "Temps de départ - hh:mm:ss.ms" in curr_df.columns:
                curr_df["Begin Time - hh:mm:ss.ms"] = curr_df["Temps de départ - hh:mm:ss.ms"]
                curr_df = curr_df.drop(columns = ["Temps de départ - hh:mm:ss.ms"])
            if 'temps de fin - hh:mm:ss.ms' in curr_df.columns:
                curr_df['End Time - hh:mm:ss.ms'] = curr_df['temps de fin - hh:mm:ss.ms']
                curr_df = curr_df.drop(columns = ['temps de fin - hh:mm:ss.ms'])
            if 'Durée - hh:mm:ss.ms' in curr_df.columns:
                curr_df['Duration - hh:mm:ss.ms'] = curr_df['Durée - hh:mm:ss.ms']
                curr_df = curr_df.drop(columns = ['Durée - hh:mm:ss.ms'])
            if ("jkakaka" in file and "1" in file) or ("8" in file and "2" in file):
                curr_df.loc[:, ["P1", "P2"]] = curr_df.loc[:, ["P2", "P1"]].values
                print("FOUND BUG") 
                print(file)
            df_time = pd.concat([df_time, curr_df])

    df_time['Time_begin_ts'] = pd.to_datetime(df_time["Begin Time - hh:mm:ss.ms"], format = "%H:%M:%S.%f")
    df_time = df_time.drop(columns = ["Unnamed: 0"])
    df_time = df_time[['Dyad', 'Session', 'Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms', 'Duration - hh:mm:ss.ms', 'P1', 'P2']]
    df_time['Time_begin_ts'] = pd.to_datetime(df_time["Begin Time - hh:mm:ss.ms"], format = "%H:%M:%S.%f")
    mask = ((df_time.Dyad == 6) & (df_time.Session == 1))
    # print(df_time.info())
    # print(type(df_time.iloc[0]["Dyad"]), type(df_time.iloc[0]["Session"]))
    # df_source.loc[mask, ["P1", "P2"]] = pd.concat([df_source.loc[mask].P2, df_source.loc[mask].P1], axis = 1)
    df_time.loc[mask, ["P1", "P2"]] = df_time.loc[mask, ["P2", "P1"]].values
    
    return df_time

if __name__ == "__main__":

    df_source, df_time = get_annotations_df().reset_index(), get_corrected_df().reset_index()

    df_fuzzy = fuzzy_match(df_source, df_time).reset_index()

    # Now, we get the maximal score found by the algorithm
    df_fuzzy["Match_score"] = df_fuzzy["Match"].apply(max_match)
    df_fuzzy = df_fuzzy.sort_values(by = "Match_score")

    df_fuzzy = df_fuzzy[["Dyad","Session","Period","Begin_time","End_time","Duration","P1","P2", "Time_begin_ts","SD","QE","SV","PR","HD","Match","Match_score"]].replace("", pd.NA)
    
    # print("Now, matching the rows with a high score")

    for idx, row in tqdm.tqdm(df_fuzzy.iterrows()):
        # Why 86? We observed that in our use case, 86 is the score corresponding to the wrong matches
        if int(row["Match_score"]) > 86:
            information = row["Match"]
            num_of_matches = len(information) - 5

            # Gather the information stocked in the fuzzymatching result
            p, matches, informations, index = information["person"], information["match"], information["informations"], information["index"]
            
            # For each match, get the index of the match in the dataframe.
            df_fuzzy.loc[idx, ["Match_index"]] = int(index[0])

    df_fuzzy_classonly = df_fuzzy.loc[df_fuzzy[["SD", "QE", "SV","PR","HD"]].dropna(thresh=1).index, :]

    # df_fuzzy.to_csv("fuzzymatch.csv")
    # df_fuzzy_classonly.to_csv("fuzzymatch_class_only.csv")

    # Now, at last, for each found index, link the annotations to the corrected dataframe

    print(df_fuzzy_classonly["Match_index"].unique())
    for idx, row in df_fuzzy_classonly.iterrows():
        idx_ = row["Match_index"]
        df_time.loc[idx_, ["SD", "QE", "SV","PR","HD"]] = row[["SD", "QE", "SV","PR","HD"]].values
    df_time = df_time[["Dyad","Session","Begin Time - hh:mm:ss.ms","End Time - hh:mm:ss.ms","Duration - hh:mm:ss.ms","P1","P2","SD","QE","SV","PR","HD"]]
    for cl_ in ["SD","QE","SV","PR","HD"]:
        print(df_time[cl_].value_counts()["x"])
    
    df_time.to_csv("Final_dataframe_2016.csv")

    if False:
        final_df = pd.DataFrame(columns = df_fuzzy.columns)

        prev_sent = ""
        prev_b_time = ""
        prev_e_time = ""

        initialized = False
        for idx, row in df_fuzzy.replace(pd.NA, "").reset_index().iterrows():
            if not initialized:
                final_df.loc[len(final_df)] = row
                previous_row = row
                initialized = True
                continue

            prev_sent = previous_row["P1"] + previous_row["P2"]
            prev_b_time = previous_row["Begin_time"]
            prev_e_time = previous_row["End_time"]

            if (prev_sent == row["P1"] + row["P2"]) and (prev_b_time == row["Begin_time"]):
                for cl_ in ["SD","QE","SV","PR","HD"]:
                    final_df.loc[len(final_df) - 1, cl_] = previous_row[cl_] + row[cl_]
            else:
                final_df.loc[len(final_df)] = row
            previous_row = row
        # df_final.loc[:, ["SD","QE","SV","PR","HD"]] = df_classes.loc[:, ["SD","QE","SV","PR","HD"]].values


        final_df = final_df.sort_values(by = ["Dyad", "Session", "Begin_time"])

        final_df_sentences = final_df.groupby(by = ["Dyad", "Session", "Period", "P1", "P2"]).first().reset_index()[["Dyad", "Session", "Period", "Begin_time", "End_time", "P1", "P2"]]
        final_df_classes = final_df.groupby(by = ["Dyad", "Session", "Period", "P1", "P2"]).sum().reset_index()[["SD","QE","SV","PR","HD"]].values
        
        # final_df.loc[:, ["Dyad", "Session", "Period", "Begin_time", "End_time", "P1", "P2"]] = final_df_sentences
        final_df_sentences[["SD","QE","SV","PR","HD"]] = final_df_classes

        final_df = final_df[["Dyad","Session","Period","Begin_time","End_time","Duration","P1","P2","SD","QE","SV","PR","HD"]]
        final_df.to_csv("final_data_2016.csv")

        # df_fuzzy_classonly = df_fuzzy.loc[df_fuzzy[["SD", "QE", "SV","PR","HD"]].dropna(thresh=1).index, :]
        # df_fuzzy_classonly.to_csv("fuzzymatch_class_only_almostcomplete.csv")