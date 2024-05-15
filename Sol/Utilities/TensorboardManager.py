import os
import tempfile
import matplotlib.pyplot as plt
from tbparse import SummaryReader

import numpy as np
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
from scipy import stats
import tensorboard as tb
import tensorflow as tf

from tensorflow.python.summary.summary_iterator import summary_iterator

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# from tbparse import SummaryReader

import time


class TBM:

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

        return tf_files

    def find_tensorflow_files_to_df(self, root_directory):
        """
        Traverses the given directory and its subdirectories to find TensorFlow files and organize them into a DataFrame.
        The DataFrame will include paths to TensorFlow files and an incremental step count based on timestamps in filenames.

        Parameters:
            root_directory (str): The path to the directory to search.

        Returns:
            DataFrame: A DataFrame containing the paths to TensorFlow files and step counts.
        """
        tf_files = []
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for file in filenames:
                if file.startswith("events.out.tfevents."):
                    timestamp = float(file.split('.')[-2])  # Extract timestamp from filename
                    full_path = os.path.join(dirpath, file)
                    tf_files.append((full_path, timestamp))
                    print(full_path)
                    print(timestamp)

        # Create DataFrame
        df = pd.DataFrame(tf_files, columns=['FilePath', 'Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        df['Step'] = range(len(df))  # Incremental step count

        return df

    def export_tensorflow_events_to_csv(self, root_directory, output_csv_path):
        """
        Corrected function to traverse directories, find TensorFlow files, and organize them into a DataFrame
        with continuous step counts across files, then export to a CSV file considering correct timestamp parsing.

        Parameters:
            root_directory (str): The path to the directory to search.
            output_csv_path (str): The path where the resulting CSV file will be saved.

        Returns:
            None: Outputs a CSV file at the specified path.
        """
        tf_files = []
        # Extract information from directories and files
        for dirpath, dirnames, filenames in os.walk(root_directory):
            # Correct timestamp extraction based on expected location in the directory name
            dir_parts = dirpath.split('/')[-1].split('.')
            if len(dir_parts) > 1 and dir_parts[1].isdigit():
                dir_timestamp = float(dir_parts[1])
            else:
                dir_timestamp = 0  # Fallback if no valid timestamp is found

            for file in filenames:
                if file.startswith("events.out.tfevents."):
                    file_parts = file.split('.')
                    if len(file_parts) > 4 and file_parts[4].isdigit():
                        file_timestamp = float(file_parts[4])
                    else:
                        file_timestamp = 0  # Fallback if no valid timestamp is found

                    # Create a combined timestamp from dir and file timestamps for more accurate sorting
                    combined_timestamp = dir_timestamp * 1e6 + file_timestamp
                    full_path = os.path.join(dirpath, file)
                    tf_files.append((full_path, combined_timestamp))

        # Create DataFrame and sort it based on the combined timestamps
        df = pd.DataFrame(tf_files, columns=['FilePath', 'CombinedTimestamp'])
        df.sort_values('CombinedTimestamp', inplace=True)

        # Simulate continuous step counts across files
        df['Step'] = range(len(df))  # Simulated step data
        # Adjust step count to be continuous across files
        max_step = 0
        for i in range(1, len(df)):
            current_max = df.loc[i - 1, 'Step'] + 10  # Assuming each file contributes an arbitrary number of 10 steps
            df.loc[i, 'Step'] += current_max  # Increment current file's start step by the max step of the previous file

        # Export to CSV
        df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':

    experiment_id = "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard"
    out = "Sol/logs/concats/PPO 05.11.2024_11.37.31.csv"
#
    tbm = TBM()
    # tbm.export_tensorflow_events_to_csv(experiment_id, out)


#     # tbm.concatenate_tf_events(experiment_id, out)
#

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
#
    f_names = tbm.find_tensorflow_files("Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard")
    print(f_names)

    reader = SummaryReader(os.path.join(experiment_id))
    df = reader.scalars
    print(df)

    dfs = pd.DataFrame()

    m_steps = 0

    for f_name in f_names:
        print(f_name)
        reader = SummaryReader(os.path.join(f_name))
        df = reader.scalars
        # if len(df) != 0:
        #     df['steps'] += m_steps
        # m_steps = df['steps'][-1]

#
#     # dfs = tbm.find_tensorflow_files_to_df(experiment_id)
#     print(dfs)

import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


# if __name__ == '__main__':
#     path = "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard"
#     to_csv(path)


    # reader = SummaryReader(os.path.join(
    #     "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard/PPO_05.11.2024_11.37.54_1/events.out.tfevents.1715420274.Ozymandias-II.18828.0"))
    #
    # df = reader.scalars
    # print(df)
    # print(reader.get_tags())

    

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
#