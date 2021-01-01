import os
import time
import torch
import pickle
import numpy as np
from model import Model
from music21 import *


def vec_to_int(v):
    return int(torch.argmax(v))


def enlarge(vector, low):
    new = torch.zeros(128)
    for i, x in enumerate(vector):
        new[i+low] = x
    return new


def get_chord(v, d, low, dec):
    d_i = vec_to_int(d)
    du = dec[d_i]
    v = enlarge(v, low)
    notes = list(map(lambda x: x[0], filter(lambda x: x[1] == 1, enumerate(v))))
    """
    notes = []
    for i, x in enumerate(v):
        if x == 1:
            notes.append(i)
    """
    if len(notes) > 0:
        c = chord.Chord(notes, duration=duration.Duration(du))
    else:
        c = note.Rest(duration=duration.Duration(du))
    return c


def turn_midi(pitch, duration, low, dec, out_file, out_tempo=75.0):
    song = stream.Stream()
    for t in range(len(pitch)):
        song.append(get_chord(pitch[t], duration[t], low, dec))
    song.append(tempo.MetronomeMark(number=out_tempo))
    song.write('midi', out_file)


with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/pitch.pkl', 'rb') as f:
    pitch = pickle.load(f)
with open('data/duration.pkl', 'rb') as f:
    duration = pickle.load(f)
with open('data/param.pkl', 'rb') as f:
    (low, dec) = pickle.load(f)

v0 = pitch[0][0]
u0 = duration[0][0]
length = 100
k = 10
out_tempo = 75.0

pitch_out, duration_out, prob = model.generate(v0, u0, length, k)

with open('music.pkl', 'wb') as f:
    pickle.dump((pitch_out, duration_out), f)
turn_midi(pitch_out, duration_out, low, dec, "music.midi", out_tempo)
