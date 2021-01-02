import os
import time
import torch
import pickle
import argparse
import music21
from model import Model


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
    if len(notes) > 0:
        c = music21.chord.Chord(notes, duration=music21.duration.Duration(du))
    else:
        c = music21.note.Rest(duration=music21.duration.Duration(du))
    return c


def turn_midi(pitch, duration, low, dec, filename, out_tempo=75.0):
    song = music21.stream.Stream()
    for t in range(len(pitch)):
        song.append(get_chord(pitch[t], duration[t], low, dec))
    song.append(music21.tempo.MetronomeMark(number=out_tempo))
    fmt = os.path.splitext(filename)[-1].lower()
    song.write(fmt, filename)


with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/pitch.pkl', 'rb') as f:
    pitch = pickle.load(f)
with open('data/duration.pkl', 'rb') as f:
    duration = pickle.load(f)
with open('data/data.pkl', 'rb') as f:
    (pitch, duration) = pickle.load(f)
with open('data/param.pkl', 'rb') as f:
    (low, dec) = pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--sample_k", type=int, default=10,
                    help="Number of rounds for Gibbs sampling")
parser.add_argument("-l", "--length", type=int, default=100,
                    help="Length of the music")
parser.add_argument("-t", "--tempo", type=float, default=75.0,
                    help="Tempo of the music")
parser.add_argument("-f", "--filename", type=str, default="music.mid",
                    help="Output filename (.mid or .xml)")
args = parser.parse_args()
for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

length = args.length
K = args.sample_k
out_tempo = args.tempo
filename = args.filename

v0 = pitch[0][0]
u0 = duration[0][0]
pitch_out, duration_out, prob = model.generate(v0, u0, length, K)

with open('music.pkl', 'wb') as f:
    pickle.dump((pitch_out, duration_out), f)

turn_midi(pitch_out, duration_out, low, dec, filename, out_tempo)

