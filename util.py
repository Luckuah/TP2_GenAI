import os
import json
import regex as re
from IPython.display import Audio
from music21 import converter, midi
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display as ipythondisplay

# --------------------------
# Fonctions de gestion des chansons
# --------------------------

def load_songs(filename):
    """Load songs from a JSON file"""
    with open(filename, 'r') as file:
        data = json.load(file)
    songs = [data[str(index)] for index in range(len(data))]
    return songs

def extract_song_snippet(text):
    """Extract songs from generated text"""
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print(f"Found {len(songs)} songs in text")
    return songs

# --------------------------
# Fonctions de conversion et lecture ABC
# --------------------------

def save_song_to_abc(song, filename="tmp"):
    """Save ABC string to a .abc file (optional)"""
    save_name = f"{filename}.abc"
    with open(save_name, "w") as f:
        f.write(song)
    return save_name

def abc2midi_music21(abc_text, midi_file="tmp.mid"):
    """Convert ABC string to MIDI using music21"""
    try:
        score = converter.parse(abc_text, format='abc')
        mf = midi.translate.music21ObjectToMidiFile(score)
        mf.open(midi_file, "wb")
        mf.write()
        mf.close()
        return midi_file
    except Exception as e:
        print("Error converting ABC to MIDI:", e)
        return None

def play_midi(midi_file):
    """Play MIDI file using pygame"""
    try:
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.quit()
        return Audio(midi_file)
    except Exception as e:
        print("Error playing MIDI:", e)
        return None

def play_song(song):
    """Play ABC string in Windows/Jupyter using music21 + pygame"""
    midi_file = abc2midi_music21(song)
    if midi_file:
        return play_midi(midi_file)
    return None

def play_generated_song(generated_text):
    """Play all valid songs from generated text"""
    songs = extract_song_snippet(generated_text)
    if len(songs) == 0:
        print("No valid songs found in generated text.")
        return

    for i, song in enumerate(songs):
        print(f"Playing song {i} ...")
        play_song(song)

def test_batch_func_types(func, args):  ##
    ret = func(*args)
    assert len(ret) == 2, "[FAIL] get_batch must return two arguments (input and label)"
    assert type(ret[0]) == np.ndarray, "[FAIL] test_batch_func_types: x is not np.array"
    assert type(ret[1]) == np.ndarray, "[FAIL] test_batch_func_types: y is not np.array"
    print("[PASS] test_batch_func_types")
    return True

def test_batch_func_shapes(func, args):  ##
    dataset, seq_length, batch_size = args
    x, y = func(*args)
    correct = (batch_size, seq_length)
    assert x.shape == correct, "[FAIL] test_batch_func_shapes: x {} is not correct shape {}".format(x.shape, correct)
    assert y.shape == correct, "[FAIL] test_batch_func_shapes: y {} is not correct shape {}".format(y.shape, correct)
    print("[PASS] test_batch_func_shapes")
    return True

def test_batch_func_next_step(func, args):  ##
    x, y = func(*args)
    assert (x[:,1:] == y[:,:-1]).all(), "[FAIL] test_batch_func_next_step: x_{t} must equal y_{t-1} for all t"
    print("[PASS] test_batch_func_next_step")
    return True

def display_model(model):
  tf.keras.utils.plot_model(model,
             to_file='tmp.png',
             show_shapes=True)
  return ipythondisplay.Image('tmp.png')


def plot_sample(x,y,vae):
    plt.figure(figsize=(2,1))
    plt.subplot(1, 2, 1)

    idx = np.where(y==1)[0][0]
    plt.imshow(x[idx])
    plt.grid(False)

    plt.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(recon[idx])
    plt.grid(False)

    plt.show()


class LossHistory:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss


class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()


def create_grid_of_images(xs, size=(5,5)):
    """ Combine a list of images into a single image grid by stacking them into an array of shape `size` """

    grid = []
    counter = 0
    for i in range(size[0]):
        row = []
        for j in range(size[1]):
          row.append(xs[counter])
          counter += 1
        row = np.hstack(row)
        grid.append(row)
    grid = np.vstack(grid)
    return grid