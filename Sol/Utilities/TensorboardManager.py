import itertools
import os
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tbparse import SummaryReader

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TBM:
    """Half of this doesn't even work."""

    def extract_datetime_from_filename(self, filename):
        match = (re.search(r'PPO_(\d{2}\.\d{2}\.\d{4}_\d{2}\.\d{2}\.\d{2})_', filename) or
                 re.search(r'PPO_save_(\d{2}\.\d{2}\.\d{4}_\d{2}\.\d{2}\.\d{2})\\', filename))
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, "%m.%d.%Y_%H.%M.%S")

    def concatenate_tf_events(self, root_dir, output_file):

        writer = tf.summary.create_file_writer(output_file)

        event_count = 0  # Initialize a counter for events

        # Walk through the directory tree
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    path = os.path.join(subdir, file.replace('\\', '/'))  # Ensure consistent path formatting
                    print(f"Processing file: {path}")
                    raw_dataset = tf.data.TFRecordDataset(path)
                    for raw_record in raw_dataset:
                        event = tf.compat.v1.Event.FromString(raw_record.numpy())
                        with writer.as_default():
                            tf.summary.experimental.write_raw_pb(event.SerializeToString(), step=0)
                        event_count += 1

        writer.close()
        print(f"Total events written: {event_count}")

    def extract_events(self, event_path):
        summaries = []
        for event in tf.compat.v1.train.summary_iterator(event_path):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    summaries.append({
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "tag": value.tag,
                        "simple_value": value.simple_value
                    })
        return pd.DataFrame(summaries)

    def read_and_concatenate_events(self, root_dir):
        all_dfs = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    path = os.path.join(subdir, file)
                    df = self.extract_events(path)
                    all_dfs.append(df)

        return all_dfs

    def write_df_to_tfevents(self, dataframe, output_path):
        # Assuming 'step' column exists and is used for indexing summaries in TensorBoard
        assert 'step' in dataframe.columns, "'step' column must be present in the DataFrame"

        # Create a TensorFlow writer
        writer = tf.summary.create_file_writer(output_path)

        # Other columns are assumed to be metrics
        metric_columns = [col for col in dataframe.columns if col != 'step']

        with writer.as_default():
            for index, row in dataframe.iterrows():
                for metric in metric_columns:
                    # Writing each metric as a scalar summary
                    tf.summary.scalar(name=metric, value=row[metric], step=int(row['step']))
                writer.flush()

    def find_tensorflow_files(self, root_directory):
        """
        Traverses the given directory and its subdirectories to find TensorFlow files.
        TensorFlow event files are identified by the prefix 'events.out.tfevents.'.

        Parameters:
            root_directory (str): The path to the directory to search.

        Returns:
            list: A list containing the paths to TensorFlow files found.
        """
        tf_files = []
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for file in filenames:
                if file.startswith("events.out.tfevents."):
                    full_path = os.path.join(dirpath, file)
                    tf_files.append(full_path)

        print(f"Found {len(tf_files)} TensorFlow files in {root_directory}")
        return tf_files

    def create_data_dict(self, merged_df):
        unique_tags = merged_df['tag'].unique()
        data_dict = {tag: [] for tag in unique_tags}

        for tag in unique_tags:
            tag_data = merged_df[merged_df['tag'] == tag]
            for _, row in tag_data.iterrows():
                data_dict[tag].append((row['step'], row['value']))

        return data_dict

    def sort_em_up(self, filenames):
        # sorted_filenames = sorted(filenames, key=self.extract_datetime_from_filename)

        unique_tags = set()
        all_data = []
        step_increment = 0

        for f_name in filenames:
            print(f"Processing file: {f_name}")
            reader = SummaryReader(f_name)
            df = reader.scalars
            if not df.empty:
                unique_tags.update(df['tag'].unique())
                df['step'] += step_increment
                step_increment = df['step'].max() + 1
                all_data.append(df)

        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"Merged DataFrame shape: {merged_df.shape}")
        return merged_df

    def visualize_dict(self, data_dict, tags_to_plot=None):
        # Plot settings
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")

        # Determine which tags to plot
        if tags_to_plot is None:
            tags_to_plot = data_dict.keys()

        # Plot each tag
        for tag in tags_to_plot:
            if tag in data_dict and data_dict[tag]:
                values = data_dict[tag]
                steps, vals = zip(*values)
                plt.plot(steps, vals, label=tag)
            else:
                print(f"Tag {tag} not found in data or has no values.")

        if plt.gca().has_data():
            # Adding titles and labels
            plt.title('Trends of Different Tags Over Steps')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        else:
            print("No data to plot.")

    def visualize_each_tag_separately(self, data_dict, output_dir=None):
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for tag, values in data_dict.items():
            if values:  # Check if values is not empty
                steps, vals = zip(*values)
                plt.figure(figsize=(12, 8))
                sns.set(style="whitegrid")
                plt.plot(steps, vals, label=tag)
                plt.title(f'Trend of {tag} Over Steps')
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.legend()
                plot_filename = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved plot for {tag} to {plot_filename}")

    def plot_runs(self, tag, runs, names, smoothing_factor=0.0):

        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        # sns.set_style("darkgrid", {"axes.facecolor": ".9"})

        for i, run in enumerate(runs):
            values = run[tag]
            steps, vals = zip(*values)

            if smoothing_factor > 0:
                smoothed_vals = self.smooth(vals, smoothing_factor)
            else:
                smoothed_vals = vals

            plt.plot(steps, smoothed_vals, label=names[i])

        # plt.title('Effects of different learning rates on ' + tag +
        # plt.title('Comparison between SAC and PPO sample efficieny on ' + tag +
        plt.title('Comparison of base and best PPO on ' + tag +
                  (" (with smoothing)") if smoothing_factor > 0 else "")
        plt.xlabel('Step')
        plt.ylabel('Value')

        if plt.gca().has_data():
            plt.legend(title="Algorithm", fontsize=20, loc='best')
            plt.show()

    def smooth(self, values, factor):
        smoothed_values = []
        for i in range(len(values)):
            if i == 0:
                smoothed_values.append(values[i])
            else:
                smoothed_values.append(smoothed_values[-1] * factor + values[i] * (1 - factor))
        return smoothed_values

    def limit_data(self, data_dict):

        for tag, values in data_dict.items():
            n_values = len(values)
            print(n_values)
            # if n_values > 1000:
            data_dict[tag] = values[:int(n_values * 0.95)]

        return data_dict

    def normalize_rewards(self, data_dict, max_reward):
        for tag, values in data_dict.items():
            data_dict[tag] = [(step, val / max_reward) for step, val in values]
        return data_dict

    def percentage_rewards(self, data_dict, max_reward):
        for tag, values in data_dict.items():
            data_dict[tag] = [(step, (val / max_reward) * 100) for step, val in values]
        return data_dict

    def limit_data_num(self, data_dict, max_items=1000):
        """
        Limit the number of items in each list within the data dictionary.

        Args:
            data_dict (dict): A dictionary where each key is associated with a list of values.
            max_items (int): The maximum number of items allowed in each list. Default is 1000.

        Returns:
            dict: The modified dictionary with each list limited to max_items elements.
        """
        for tag, values in data_dict.items():
            n_values = len(values)
            print(n_values)
            if n_values > max_items:
                data_dict[tag] = values[:max_items]

        return data_dict


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']
    print(tags)
    for it in summary_iterators:
        print(it.Tags())
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


# with np.load(os.path.join(experiment_id + "evaluations.npz")) as data:
#     for key in data.keys():
#         print(key)

# df = tf.data.TFRecordDataset(os.path.join( experiment_id + "evaluations.npz"), compression_type='npz')
# print(df)
#
# raw_example = next(iter(df))
# parsed = tf.train.Example.FromString(raw_example.numpy())
# print(raw_example)
# print(parsed)
#
#
# for e in summary_iterator(experiment_id + "evaluations.npz"):
#   for v in e.summary.value:
#     if v.tag == 'y=2x':
#       print(e.step, v.simple_value)


# def tabulate_events(dpath):
#     summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
#
#     tags = summary_iterators[0].Tags()['scalars']
#
#     for it in summary_iterators:
#         assert it.Tags()['scalars'] == tags
#
#     out = {tag: [] for tag in tags}
#     steps = []
#
#     for tag in tags:
#         steps = [e.step for e in summary_iterators[0].Scalars(tag)]
#
#         for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
#             assert len(set(e.step for e in events)) == 1
#
#             out[tag].append([e.value for e in events])
#
#     return out, steps
#
# sdict, steps = tabulate_events(experiment_id)
#
# print(sdict, steps)

# dfw = experiment.get_scalars(pivot=True)
# print(dfw)


if __name__ == '__main__':

    # experiment_id = "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard"

    tbm = TBM()

    # reader = SummaryReader(experiment_id)
    # df = reader.tensors
    # print(df)
    # print(SummaryReader(experiment_id, pivot=True).scalars)
    #
    #     # run_dir = os.path.join(log_dir, 'run0')
    #
    #     # dfs = tbm.read_and_concatenate_events(experiment_id)
    #     #
    #     # for df in dfs:
    #     #     print(df)
    #     #     tbm.write_df_to_tfevents(df, out)

    # # f_names = tbm.find_tensorflow_files("Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard/")
    # f_names = tbm.find_tensorflow_files("Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard/PPO_05.11.2024_11.37.54_1")
    # f_names2 = tbm.find_tensorflow_files("Sol/logs/PPO_save_05.15.2024_00.03.17")
    #
    # print(f_names)
    #
    # sorted_filenames = sorted(f_names, key=tbm.extract_datetime_from_filename)
    # print(f_names)
    #
    # l = []
    # for f_name in f_names:
    #     l.append(SummaryReader(os.path.join(f_name)).scalars)
    #     print(f"Checking file: {f_name}")
    #     if os.path.exists(f_name):
    #         print(f"File exists: {f_name}")
    #     else:
    #         print(f"File does NOT exist: {f_name}")
    # print(l)
    #
    # merged_df = tbm.sort_em_up(f_names)
    # # merged_df.to_csv("Sol/visual/based.csv", index=False)
    # data_dict = tbm.create_data_dict(merged_df)

    tags = [
        # "eval/mean_ep_length",
        "eval/mean_reward"
    ]
    # tbm.visualize_dict(data_dict, tags_to_plot=tags)
    # tbm.visualize_each_tag_separately(data_dict, output_dir="Sol/logs/concats")

    # f_names = ["Sol/logs/PPO 05.13.2024_20.04.44/PPO_2",
    #
    #            ]
    # nn size
    # tbm.plot_runs("eval/mean_reward", [tbm.create_data_dict(tbm.sort_em_up(f_name)) for f_name in f_names],
    #               names=["256", "512"])

    # #batch
    # f_names = ["Sol/logs/PPO 05.13.2024_20.04.44/PPO_2",
    #            "Sol/logs/PPO_save_05.16.2024_09.37.34/PPO_1",
    #            "Sol/logs/PPO_save_05.17.2024_01.36.12/PPO_1"
    #            ]
    # tbm.plot_runs("eval/mean_reward",
    #               [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))) for f_name in f_names],
    #               names=["256", "512", "1024"])

    # clip range
    # f_names = [
    #     # "Sol/logs/PPO 05.13.2024_20.04.44/PPO_2",
    #     #        "Sol/logs/PPO_save_05.15.2024_00.03.17/PPO_1",
    #            "Sol/logs/PPO_save_05.17.2024_12.25.36/PPO_1",
    #            "Sol/logs/PPO_save_05.17.2024_16.18.07/PPO_1"]
    #
    # tbm.plot_runs("eval/mean_reward", [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name)))
    #                                    for f_name in f_names],
    #               names=[
    #                   # "0.1", "0.2",
    #                   "0.35, lr=3.e-4", "0.35, lr=2.5e-4"])
    # tbm.plot_runs("train/approx_kl", [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name)))
    #                                    for f_name in f_names],
    #               names=[
    #                   # "0.1", "0.2",
    #                   "0.35, lr=3.e-4", "0.35, lr=2.5e-4"])
    # tbm.plot_runs("train/clip_fraction", [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name)))
    #                                    for f_name in f_names],
    #               names=[
    #                   # "0.1", "0.2",
    #                   "0.35, lr=3.e-4", "0.35, lr=2.5e-4"])

    # # gamma
    # f_names = [
    #            "Sol/logs/PPO_save_05.16.2024_09.37.34/PPO_1",
    #            "Sol/logs/PPO_save_05.19.2024_15.36.44/PPO_1/",
    #            ]
    # tbm.plot_runs("eval/mean_reward",
    #               [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))) for f_name in f_names],
    #               names=["0.99", "0.999"])

    # norm
    # f_names = ["Sol/logs/PPO_save_05.17.2024_20.23.13",
    #            "Sol/logs/PPO_save_05.18.2024_17.04.26"
    #
    #            ]
    # tbm.plot_runs("eval/mean_reward",
    #               [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))) for f_name in f_names],
    #               names=["0.99", "0.999"])

    # # Run 1 (max reward 600)
    # f_names1 = tbm.find_tensorflow_files("Sol/logs/PPO_save_05.19.2024_15.36.44/")
    # # Run 2 (max reward 10)
    # f_names2 = tbm.find_tensorflow_files("Sol/logs/PPO_save_05.17.2024_20.23.13")
    #
    # # Process and normalize data
    # data_dict1 = tbm.create_data_dict(tbm.sort_em_up(f_names1))
    # data_dict2 = tbm.create_data_dict(tbm.sort_em_up(f_names2))
    #
    # for i, data in enumerate(data_dict1["eval/mean_reward"]):
    #     print(data)
    #     data_dict1["eval/mean_reward"][i] = (data[0], data[1] / 60)
    #     print(data_dict1["eval/mean_reward"][i])
    # # Compare runs using normalized rewards
    # tbm.plot_runs("eval/mean_reward",
    #               [data_dict1, data_dict2],
    #               names=["Without reward normalization, scaled", "Reward normalized"])

    # # Compare runs using percentage rewards
    # tbm.plot_runs("eval/mean_reward",
    #               [data_dict_percentage_1, data_dict_percentage_2],
    #               names=["0.99", "0.999"])

    # n epoch
    # f_names = ["Sol/logs/PPO_save_05.20.2024_13.28.07/",
    #            "Sol/logs/PPO_save_05.20.2024_18.38.56/PPO_1",
    #            "Sol/logs/PPO_save_05.19.2024_23.11.04/PPO_1",
    #            "Sol/logs/PPO_save_05.19.2024_15.36.44/"
    #            ]
    # tbm.plot_runs("eval/mean_reward",
    #               [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))) for f_name in f_names],
    #               names=["3", "5", "10", "20"])
    # lr
    # f_names = [
    #     "Sol/logs/PPO_save_05.19.2024_23.11.04/PPO_1",
    #     "Sol/logs/PPO_save_05.23.2024_18.13.59",
    #     "Sol/logs/PPO_save_05.23.2024_12.22.23",
    #     "Sol/logs/PPO_save_05.23.2024_16.10.52",
    #     "Sol/logs/PPO_save_05.25.2024_18.40.52"
    # ]
    # tbm.plot_runs("eval/mean_reward",
    #               [tbm.limit_data(tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name)))) for f_name in
    #                f_names],
    #               names=["2.5e-4", "5e-4", "1e-3", "5e-3",
    #                      "linear decay from 2.5e-4 to 0"],
    #               smoothing_factor=0.75)

    # target kl, value clip, entropy

    # f_names = [
    #     "Sol/logs/PPO_save_05.24.2024_18.39.12",
    #     "Sol/logs/PPO_save_05.24.2024_21.27.54",
    #     "Sol/logs/PPO_save_05.25.2024_02.21.23",
    #     "Sol/logs/PPO_save_05.25.2024_15.52.28",
    # ]
    # for tag in ["eval/mean_reward", "train/approx_kl", "train/value_loss"]:
    #     tbm.plot_runs(tag,
    #                   [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))) for f_name in
    #                    f_names],
    #                   names=["Only target_kl=0.05",
    #                          "with vf_clip=0.3",
    #                          "with vf_clip=0.3, entropy=0.01",
    #                          "with vf_clip=0.3, entropy=0.1"],
    #                   smoothing_factor=0.7)

    # sac
    # f_names = ["Sol/logs/SAC_save_05.26.2024_00.12.30"
    #            ]
    # tags = [
    #     # "eval/mean_ep_length",
    #     "eval/mean_reward",
    #     "train/critic_loss",
    #     "train/ent_coef",
    # ]
    #
    # for tag in tags:
    #     tbm.plot_runs(tag,
    #                   [tbm.limit_data(tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))))
    #                    for f_name in f_names],
    #                   names=[""],
    #                   smoothing_factor=0.8)

    # sac2

    # f_names = ["Sol/logs/SAC_save_05.21.2024_23.28.56",
    #            os.path.join(
    #                "Sol/logs/PPO 05.11.2024_11.37.31\ppo_tensorboard\PPO_05.11.2024_11.37.54_1"),
    #            ]
    # tags = [
    #     # "eval/mean_ep_length",
    #     "eval/mean_reward",
    #     # "train/critic_loss",
    #     # "train/ent_coef",
    # ]
    #
    # for tag in tags:
    #     tbm.plot_runs(tag,
    #                   [tbm.limit_data_num(tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name))),
    #                                       203)
    #                    for f_name in f_names],
    #                   names=["SAC", "PPO"],
    #                   smoothing_factor=0.8)

    f_names = ["Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard/",
               # "Sol/logs/PPO_save_05.25.2024_02.21.23",
               "Sol/logs/PPO_save_05.25.2024_18.40.52",
               ]
    tags = [
        # "eval/mean_ep_length",
        "eval/mean_reward",
        # "train/critic_loss",
        # "train/ent_coef",
    ]

    for tag in tags:
        tbm.plot_runs(tag,
                      [tbm.create_data_dict(tbm.sort_em_up(tbm.find_tensorflow_files(f_name)))
                       for f_name in f_names],
                      names=["Base", "Best", ""],
                      smoothing_factor=0.5)
