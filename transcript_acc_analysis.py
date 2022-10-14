import pandas as pd
import faulthandler
import numpy as np

# from numpy_ext import rolling_apply


def main1():
    """

    Returns:
        _type_: _description_
    """
    df = pd.read_csv("dummy_csv.csv")

    nvb_list = [
        "NVB1_P2",
        "NVB2_P2",
        "NVB3_P2",
        "NVB1_P1",
        "NVB4_P1",
        "NVB5_P1",
        "NVB3_P1",
        "NVB5_P2",
        "NVB4_P2",
        "NVB2_P1",
    ]

    def time_to_ms(datetime):
        if len(datetime) == 12:
            l_datetime = (
                datetime.split(":")[:-1]
                + [datetime.split(".")[0].split(":")[-1]]
                + [datetime.split(".")[-1]]
            )
            ms_time = int(l_datetime[-1]) + 1000 * (
                int(l_datetime[-2])
                + 60 * (int(l_datetime[-3]) + 60 * int(l_datetime[-4]))
            )
            return ms_time
        elif len(datetime) == 9:
            l_datetime = [
                datetime.split(":")[0],
                datetime.split(":")[1].split(".")[0],
                datetime.split(".")[-1],
            ]
            ms_time = int(l_datetime[-1]) + 1000 * (
                int(l_datetime[-2]) + 60 * (int(l_datetime[-3]))
            )
            return ms_time
        elif len(datetime) == 10:
            l_datetime = (
                datetime.split(":")[:-1]
                + [datetime.split(".")[0].split(":")[-1]]
                + [datetime.split(".")[-1]]
            )
            ms_time = 100 * int(l_datetime[-1]) + 1000 * (
                int(l_datetime[-2])
                + 60 * (int(l_datetime[-3]) + 60 * int(l_datetime[-4]))
            )
            return ms_time
        elif len(datetime) == 7:
            l_datetime = [
                datetime.split(":")[0],
                datetime.split(":")[1].split(".")[0],
                datetime.split(".")[-1],
            ]
            ms_time = 100 * int(l_datetime[-1]) + 1000 * (
                int(l_datetime[-2]) + 60 * (int(l_datetime[-3]))
            )
            return ms_time

    # df["Time_Start"] = df["Time_Start"].apply(lambda hr : pd.Timestamp(year = 2020, month=1, day=1, hour=12, minute = int(hr.split(":")[0]), second = int(hr.split(":")[1].split(".")[0]), microsecond = int(hr.split(":")[1].split(".")[1]+"00000")))
    # df["Time_End"] = df["Time_End"].apply(lambda hr : pd.Timestamp(year = 2020, month=1, day=1, hour=12, minute = int(hr.split(":")[0]), second = int(hr.split(":")[1].split(".")[0]), microsecond = int(hr.split(":")[1].split(".")[1]+"00000")))

    df["Time_Start"] = pd.to_datetime(df["Time_Start"], format="%M:%S.%f")
    df["Time_End"] = pd.to_datetime(df["Time_End"], format="%M:%S.%f")

    # df["Time_start_ms"] = df["Time_Start"].apply(time_to_ms)
    # df["Time_end_ms"] = df["Time_End"].apply(time_to_ms)

    # df["Time_Start"] = df["Time_Start"].apply(lambda hr : pd.to_datetime(hr, format = r"%M:%S.%f"))

    ####################### Get the matrix representing the context
    # Series of the utterances

    def series_to_matrix(series):

        time_start = series.iloc[0][0]

        # time_steps_list = [
        #     time_start_context + i * time_step
        #     for i in range(int(window_length / time_step))
        # ]

        time_steps = np.linspace(
            time_start, time_start + pd.Timedelta(seconds=7), 14
        )  # Remplis avec la doc
        nvb_context = np.array(shape=(len(time_steps), len(nvb_list)))

        df_nvb = pd.DataFrame.from_items(zip(series.index, series.values))

        for i_, ts_ in enumerate(time_steps):  # time_steps de size 14
            curr_df_ = df_nvb[
                (df_nvb[df_nvb.columns[0]] < ts_) & (df_nvb[df_nvb.columns[1]] > ts_)
            ]
            curr_df = curr_df[nvb_list]
            result = curr_df.any().to_numpy()
            nvb_context[i_] = result

        return 1

    for col in nvb_list:
        df[col] = df[col].replace("x", True).fillna(False).astype(int)

    df = df.sort_values(by="Time_Start", ascending=True)

    df["Informations"] = list(df[["Time_Start", "Time_End"] + nvb_list].values)

    print(df["Informations"])

    test = (
        df.groupby(["Dyad", "Session", "Period"], dropna=False, as_index=False)[
            ["Time_Start", "Informations"]
        ]
        .rolling(window="7s", on="Time_Start")["Informations"]
        .apply(series_to_matrix)
    )
    print(test)

    # print(type(series), len(series))

    # Create column [t_0, ${NVB_i}_{i}$]


def duration_offset():
    """
    duration_offset
    This function uses goes through a dataset having columnms ["Dyad", "Session", "Time_begin", "Time_end", "Time_begin_true", "Time_end_true"]

    The path to the file is not an argument, because of laziness.
    Using our hand-made alignement check, this script computes the graphs of the utterance duration errors along the script.
    (i.e. if a sentence is labelled as happening from 15:00 to 16:00, but in the video it occurs from 21:00 to 22:00, this is an outrageous mistake but this script will not pick it -> same duration)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df_2016 = pd.read_csv("data/2016_important/dyads_1_9_5perc_samples.csv")[
        [
            "Dyad",
            "Session",
            "Period",
            "Time_begin",
            "Time_end",
            "Duration",
            "P1",
            "P2",
            "Duration_s",
            "Time_begin_true",
            "Time_end_true",
            "Video",
        ]
    ]

    ts_list = ["Time_begin", "Time_end", "Time_begin_true", "Time_end_true"]

    for col_ in ts_list:
        try:
            df_2016[col_] = pd.to_datetime(df_2016[col_])
        except:
            print(col_)

    # Pretty straightforward manipulations

    df_2016["Duration"] = pd.to_datetime(df_2016["Duration"]).dt.second

    df_2016["Time_begin_offset"] = (
        df_2016["Time_begin"].dt.second
        - df_2016["Time_begin_true"].dt.second
        + 60 * (df_2016["Time_begin"].dt.minute - df_2016["Time_begin_true"].dt.minute)
    )
    df_2016["Time_end_offset"] = (
        df_2016["Time_end"].dt.second
        - df_2016["Time_end_true"].dt.second
        + 60 * (df_2016["Time_end"].dt.minute - df_2016["Time_end_true"].dt.minute)
    )
    df_2016["Duration_true"] = (
        df_2016["Time_end_true"].dt.second
        - df_2016["Time_begin_true"].dt.second
        + 60
        * (df_2016["Time_end_true"].dt.minute - df_2016["Time_begin_true"].dt.minute)
    )
    df_2016["Duration_error"] = df_2016["Duration"] - df_2016["Duration_true"]

    print(len(df_2016["Dyad"].unique()))

    # Set up figure
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None) #Adjust space between graphs

    for i, dyad_ in enumerate(df_2016["Dyad"].dropna().unique()):
        for j, session_ in enumerate(df_2016["Session"].dropna().unique()):
            curr_df_ = df_2016[
                (df_2016["Dyad"] == dyad_) & (df_2016["Session"] == session_)
            ]
            begin_median_value = curr_df_["Time_begin_offset"].quantile(0.5)
            begin_75_value = curr_df_["Time_begin_offset"].quantile(0.75)
            begin_25_value = curr_df_["Time_begin_offset"].quantile(0.25)
            end_median_value = curr_df_["Time_end_offset"].quantile(0.5)
            end_75_value = curr_df_["Time_end_offset"].quantile(0.75)
            end_25_value = curr_df_["Time_end_offset"].quantile(0.25)
            with open("offset_sumup.txt", "a") as f:
                f.write(
                    "\n========  DYAD {} SESSION {}  ========\n".format(
                        int(dyad_), int(session_)
                    )
                )
                f.write("--- BEGIN TIMES :---\n")
                f.write(
                    "25 centile {}\n50 centile {}\n75 centile {}\n".format(
                        begin_25_value, begin_median_value, begin_75_value
                    )
                )
                f.write("----- END TIMES :---\n")
                f.write(
                    "25 centile {}\n50 centile {}\n75 centile {}\n".format(
                        end_25_value, end_median_value, end_75_value
                    )
                )
            plt.figure(4 * i + 2 * j)
            plt.ylim(-50, 50)
            plt.xlim(pd.to_datetime("00:00:00"), pd.to_datetime("00:59:59"))
            if dyad_ == 0:
                continue
            first_ts = list(curr_df_["Time_begin_true"])[0]
            last_ts = list(curr_df_["Time_end_true"])[-1]
            # x_axis = pd.date_range(first_ts, last_ts)
            # y_axis1 = curr_df_["Duration_error"]
            # y_axis2 = curr_df_["Time_begin_offset"]
            # y_axis3 = curr_df_["Time_end_offset"]

            # X axis is the timestamps, Y axis is the offset in seconds

            plt.bar(
                curr_df_["Time_begin_true"],
                curr_df_["Time_begin_offset"],
                width=0.0004,
                color="blue",
                alpha=0.5,
                label="Beginning time offset",
            )
            plt.bar(
                curr_df_["Time_begin_true"],
                curr_df_["Time_end_offset"],
                width=0.0004,
                color="green",
                alpha=0.5,
                label="Ending time offset",
            )
            title_ = "Dyad {} Session {}".format(int(dyad_), session_)
            plt.ylabel("Seconds of offset")
            plt.xlabel("Timestamp")
            plt.title(title_)
            plt.legend()
            plt.savefig(
                "data/2016_important/offset/dyad_{}_session_{}_offset.jpg".format(
                    int(dyad_), session_
                )
            )

            plt.figure(4 * i + 2 * j + 1)
            plt.ylim(-50, 50)
            plt.xlim(pd.to_datetime("00:00:00"), pd.to_datetime("00:59:59"))
            plt.bar(
                curr_df_["Time_begin_true"],
                curr_df_["Duration_error"],
                width=0.0004,
                color="red",
                alpha=1,
                label="Duration error",
            )
            title_ = "Dyad {} Session {}".format(int(dyad_), session_)
            plt.ylabel("Seconds of duration offset")
            plt.xlabel("Timestamp")
            plt.title(title_)
            plt.legend()
            plt.savefig(
                "data/2016_important/offset/dyad_{}_session_{}_duration_offset.jpg".format(
                    int(dyad_), session_
                )
            )
    plt.cla()
    # plt.show()


def transcript_errors():
    """
    This function goes through a dataset having columns ["Dyad", "Session", "Time_begin", "Time_end", "Time_begin_true", "Time_end_true"]
    It simply plots in green (resp. blue) the error between true timestamp and observed timestamp.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df_2016 = pd.read_csv("data/2016_important/dyads_1_9_5perc_samples.csv")[
        [
            "Dyad",
            "Session",
            "Period",
            "Time_begin",
            "Time_end",
            "Duration",
            "P1",
            "P2",
            "Duration_s",
            "Time_begin_true",
            "Time_end_true",
            "Video",
            "Error",
        ]
    ]

    ts_list = ["Time_begin", "Time_end", "Time_begin_true", "Time_end_true"]

    for col_ in ts_list:
        try:
            df_2016[col_] = pd.to_datetime(df_2016[col_])
        except:
            print(col_)

    df_2016["Duration"] = pd.to_datetime(df_2016["Duration"]).dt.second

    print(len(df_2016["Dyad"].unique()))

    # Set up figure
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None) #Adjust space between graphs

    for i, dyad_ in enumerate(df_2016["Dyad"].dropna().unique()):
        for j, session_ in enumerate(df_2016["Session"].dropna().unique()):
            plt.figure(56 + 2 * i + j)
            plt.ylim(0, 2)
            plt.xlim(pd.to_datetime("00:00:00"), pd.to_datetime("00:59:59"))
            if dyad_ == 0:
                continue
            curr_df_ = df_2016[
                (df_2016["Dyad"] == dyad_) & (df_2016["Session"] == session_)
            ]
            # x_axis = pd.date_range(first_ts, last_ts)
            # y_axis1 = curr_df_["Duration_error"]
            # y_axis2 = curr_df_["Time_begin_offset"]
            # y_axis3 = curr_df_["Time_end_offset"]
            plt.bar(
                curr_df_["Time_begin"],
                curr_df_["Error"],
                width=0.0003,
                color="red",
                alpha=1,
                label="Wrong annotation / bad audio quality",
            )
            test = -1 * curr_df_["Error"] + 1
            plt.bar(
                curr_df_["Time_begin"],
                test,
                width=0.0003,
                color="green",
                alpha=0.3,
                label="Good annotation",
            )
            title_ = "Dyad {} Session {}".format(int(dyad_), int(session_))
            plt.ylabel("Bad quality data")
            plt.xlabel("Timestamp")
            plt.title(title_)
            plt.legend()
            plt.savefig(
                "data/2016_important/error/dyad_{}_session_{}_error.jpg".format(
                    int(dyad_), session_
                )
            )
    # plt.show()


def openface_confidence():
    """
    This function goes through a dataset having columns ["Dyad", "Session", "Error":int(bool)]. It plots the moments where an  was not present in utterance from the transcript was not present in the video.
    It simply plots in red the timestamps where it happened.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df_aus = pd.read_csv("data/2016_important/2016_aus_med_500ms_conf.csv")[
        ["dyad", "session", "participant", "timestamp", "confidence"]
    ]
    for i, dyad_ in enumerate(df_aus["dyad"].dropna().unique()):
        if int(dyad_) in [10, 11, 12]:
            for j, session_ in enumerate(df_aus["session"].dropna().unique()):
                for k, participant_ in enumerate(
                    df_aus["participant"].dropna().unique()
                ):
                    plt.figure(4 * i + 2 * j + k)
                    plt.ylim(0, 2)
                    plt.xlim(0, 4000)
                    curr_df_ = df_aus[
                        (df_aus["dyad"] == dyad_)
                        & (df_aus["session"] == session_)
                        & (df_aus["participant"] == participant_)
                    ]
                    metric_ = len(curr_df_)
                    if metric_ > 10:
                        inverted_conf_ = [1 - x for x in curr_df_["confidence"]]
                        plt.bar(
                            curr_df_["timestamp"],
                            inverted_conf_,
                            color="red",
                            width=8000 / (1 + len(curr_df_)),
                            label="Face detection confidence",
                        )
                        plt.ylabel("Face detection condfidence")
                        plt.xlabel("Timestamp")
                        title_ = "Dyad {} Session {} Participant".format(
                            int(dyad_), int(session_), int(participant_)
                        )
                        plt.title(title_)
                        plt.savefig(
                            "data/2016_important/confidence/dyad_{}_session_{}_participant_{}_confidence.jpg".format(
                                int(dyad_), session_, int(participant_)
                            )
                        )


if __name__ == "__main__":
    duration_offset()
    transcript_errors()
    openface_confidence()
