from os import listdir
from os.path import isfile, join
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

from google.protobuf.json_format import MessageToDict

def find_first_file_in(experiment_filename, platform_name, stage_name):
  path = f'{experiment_filename}/{platform_name}/{stage_name}'
  print(path)
  return f'{path}/{listdir(path)[0]}'

def event_simple_value_decoder(event_value):
  event_as_dict = MessageToDict(event_value)
  return event_as_dict['simpleValue']

def event_tensor_value_decoder(event_value):
  decoded_tensor = tf.io.decode_raw( event_value.tensor.tensor_content , tf.float32)
  return decoded_tensor.numpy()[0]

def decode_event(record_bytes, metrics, value_decoder):

  event = tf.compat.v1.Event.FromString(record_bytes.numpy())
  if( 'summary' in MessageToDict(event).keys() ):

    event_value = event.summary.value[0]
    if( event_value.tag in metrics.keys() ):

      if( event_value.tag == 'epoch_loss'):
        metrics['wall_time'].append( event.wall_time )
        metrics['time'].append( event.wall_time - metrics['wall_time'][0] )

      metrics[event_value.tag].append( value_decoder(event_value) )


def parse_record_file(file_name, value_decoder):

  parsed_metrics = {
    'epoch_loss' : [],
    'epoch_sparse_categorical_accuracy' : [],
    'epoch_steps_per_second' : [],
    'wall_time' : [],
    'time' : []
  }

  for raw_record in tf.data.TFRecordDataset(file_name):
    decode_event(raw_record, parsed_metrics, value_decoder)

  return parsed_metrics

def parse_tensorboard_logs(logs_path):
  def _parse_record_file(logs_path, platform, stage, value_decoder):
    full_path = find_first_file_in(logs_path, platform,  stage)
    return parse_record_file(full_path, value_decoder)

  return (
    _parse_record_file(logs_path, 'pharo',  'train',      event_simple_value_decoder),
    _parse_record_file(logs_path, 'pharo',  'validation', event_simple_value_decoder),
    _parse_record_file(logs_path, 'vast',   'train',      event_simple_value_decoder),
    _parse_record_file(logs_path, 'vast',   'validation', event_simple_value_decoder),
    _parse_record_file(logs_path, 'python', 'train',      event_tensor_value_decoder),
    _parse_record_file(logs_path, 'python', 'validation', event_tensor_value_decoder)
  )

markers = {
  'pharo': 'x',
  'vast': '+',
  'python': 'o'
}

def plot_training_curves(experiment, save_path=None):
  """
  Line plots: Loss and Accuracy side by side
  Solid line = training, dashed line = validation
  One shared legend for both charts
  """
  fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  platform_colors = {platform: color_cycle[i % len(color_cycle)]
                      for i, platform in enumerate(experiment['metrics'].keys())}

  metric_names = ["epoch_loss", "epoch_sparse_categorical_accuracy"]
  labels = ["Loss", "Accuracy"]

  handles, legend_labels = None, None

  for ax, metric_name, label in zip(axes, metric_names, labels):
    for platform, data in experiment['metrics'].items():
      train_values = data["train"][metric_name]
      val_values = data["validation"][metric_name]
      epochs = np.arange(1, len(train_values) + 1)

      color = platform_colors[platform]
      marker = markers.get(platform, None)

      ax.plot(
        epochs, train_values,
        linestyle="-", marker=marker, color=color,
        label=f"{platform} - train"
      )
      ax.plot(
        epochs, val_values,
        linestyle="--", marker=marker, color=color,
        label=f"{platform} - val"
      )

    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Capture legend handles only once (from the first subplot)
    if handles is None:
      handles, legend_labels = ax.get_legend_handles_labels()

  fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1))
  plt.tight_layout(rect=[0, 0, 1, 0.95])

  if save_path:
    fig.savefig(f"{experiment['folder']}/{save_path}")
    plt.close(fig)
  else:
    plt.show()

def plot_epoch_time(experiment, save_path=None):
  """
  Line plot: epoch duration (time per epoch)
  """
  plt.figure(figsize=(8, 5))

  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  platform_colors = {platform: color_cycle[i % len(color_cycle)]
                      for i, platform in enumerate(experiment['metrics'].keys())}

  for platform, data in experiment['metrics'].items():
    times = data["train"]["time"]  # duration per epoch
    epochs = np.arange(1, len(times) + 1)

    plt.plot(
      epochs, times,
      linestyle="-", marker=markers.get(platform, None),
      color=platform_colors[platform],
      label=f"{platform}"
    )

  plt.title(f"{experiment['name']} - Epoch duration")
  plt.xlabel("Epoch")
  plt.ylabel("Seconds")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.7)

  plt.tight_layout()
  if save_path: 
    plt.savefig(f"{experiment['folder']}/{save_path}")
    plt.close()
  else:
    plt.show()


def bar_total_training_time(save_path=None):
  """
  Bar chart: total training time per experiment per platform
  """
  platforms = list(experiments[0]['metrics'].keys())
    
  x = np.arange(len(experiments))
  width = 0.2

  plt.figure(figsize=(8, 5))
  for i, platform in enumerate(platforms):
    vals = []
    for exp in experiments:
      vals.append(exp['metrics'][platform]["train"]["time"][-1])
    plt.bar(x + i * width, vals, width, label=platform)

  plt.title("Total training time per experiment")
  plt.ylabel("Total time (s)")
  plt.xticks(x + width, [ exp['name'] for exp in experiments ])
  plt.legend()
  plt.grid(axis="y", linestyle="--", alpha=0.7)
  
  plt.tight_layout()
  if save_path: 
    plt.savefig(save_path)
    plt.close()
  else:
    plt.show()

if __name__ == '__main__':
  experiments = [
    { 'folder': './logs/experiment-1', 'name': 'Experiment 1', 'metrics': {} },
    { 'folder': './logs/experiment-2', 'name': 'Experiment 2', 'metrics': {} },
    { 'folder': './logs/experiment-3', 'name': 'Experiment 3', 'metrics': {} },
  ]

  for experiment in experiments:
    (
      pharo_train_metrics,
      pharo_val_metrics,
      vast_train_metrics,
      vast_val_metrics,
      python_train_metrics,
      python_val_metrics
    ) = parse_tensorboard_logs(experiment['folder'])
    
    experiment['metrics'] = {
      'pharo' : { 'train': pharo_train_metrics, 'validation': pharo_val_metrics },
      'vast' : { 'train': vast_train_metrics, 'validation': vast_val_metrics },
      'python' : { 'train': python_train_metrics, 'validation': python_val_metrics },
    }

  plot_training_curves(experiments[0], 'training-curves.png')
  plot_epoch_time(experiments[0], 'training-times.png')

  plot_training_curves(experiments[1], 'training-curves.png')
  plot_epoch_time(experiments[1], 'training-times.png')

  plot_training_curves(experiments[2], 'training-curves.png')
  plot_epoch_time(experiments[2], 'training-times.png')

  bar_total_training_time(save_path='total-training-time.png')