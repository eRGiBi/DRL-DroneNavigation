import os
import tempfile

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

    def measure_time(f):

        def timed(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()

            # print '%r (%r, %r) %2.2f sec' % \
            #       (f.__name__, args, kw, te-ts)
            return result

        return timed

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


if __name__ == '__main__':
    major_ver, minor_ver, _ = version.parse(tb.__version__).release
    print("TensorBoard version: ", tb.__version__)

    experiment_id = "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard"
    out = "Sol/logs/PPO 05.11.2024_11.37.31/ppo_tensorboard/events.out.tfevents.concatenated"

    tbm = TBM()
    tbm.concatenate_tf_events(experiment_id, out)



    # experiment_id = "Sol/model_chkpts/save-05.05.2024_20.07.35/best_model.zip"
    experiment_id = "Sol/model_chkpts/save-05.05.2024_20.07.35/"

    tup = ["Sol/model_chkpts/save-05.11.2024_11.37.31/best_model.zip", "Sol/model_chkpts/save-05.11.2024_17.33.04/best_model.zip"]





    reader = SummaryReader(experiment_id)

    # reader = SummaryReader(experiment_id, extra_columns={'dir_name'})
    df = reader.tensors
    print(df)
    print(SummaryReader(experiment_id, pivot=True).scalars)

tmpdirs = {}
tmpdirs['tensorboardX'] = tempfile.TemporaryDirectory()
log_dir = tmpdirs['tensorboardX'].name

run_dir = os.path.join(log_dir, 'run0')



with np.load(os.path.join( experiment_id + "evaluations.npz")) as data:

    for key in data.keys():
        print(key)




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


# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
# print(df)

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
# csv_path = '/tmp/tb_experiment_1.csv'
# dfw.to_csv(csv_path, index=False)
# dfw_roundtrip = pd.read_csv(csv_path)
# pd.testing.assert_frame_equal(dfw_roundtrip, dfw)
#
# # Filter the DataFrame to only validation data, which is what the subsequent
# # analyses and visualization will be focused on.
# dfw_validation = dfw[dfw.run.str.endswith("/validation")]
# # Get the optimizer value for each row of the validation DataFrame.
# optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0])
#
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# sns.lineplot(data=dfw_validation, x="step", y="epoch_accuracy",
#              hue=optimizer_validation).set_title("accuracy")
# plt.subplot(1, 2, 2)
# sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
#              hue=optimizer_validation).set_title("loss")
