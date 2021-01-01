import os
import time
import torch
from music21 import *
import numpy as np
import pickle


def reduce(vector, high, low):
    new = torch.zeros(high-low+1)
    for i in range(high-low+1):
        new[i] = vector[i+low]
    return new


def enlarge(vector, low):
    new = torch.zeros(128)
    for i, x in enumerate(vector):
        new[i+low] = 1.0


def read_chord(p):
    vector = torch.zeros(128)
    for n in p.pitches:
        vector[n.midi] = 1.0
    return vector


def read_note(p):
    vector = torch.zeros(128)
    vector[p.midi] = 1.0
    return vector


def read_rest(p):
    vector = torch.zeros(128)
    return vector


def int_to_vec(i, len):
    new = torch.zeros(len)
    new[i] = 1.0
    return new


def vec_to_int(v):
    return torch.argmax(v, dim=1)


def read_midi(path='data/midi'):
    # environment.UserSettings()['lilypondPath'] = 'C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe'
    all_songs, all_keys, all_durations, all_notes = [], [], [], []
    # Get all songs in list all_songs
    for o in os.listdir(path):
        file = os.path.join(path, o)
        print("Parsing {}".format(file))
        s = converter.parse(file)
        cs = s.chordify()
        all_songs.append(cs)

    # Preprocess all_songs to get the encodings for all durations and the corresponding table.
    # Also the highest and lowest pitch
    for i, song in enumerate(all_songs):
        all_keys.append(str(song.analyze('key')))
        for p in song:
            # print(type(p.duration.quarterLength))
            if isinstance(p, note.Note):
                all_durations.append(p.duration.quarterLength)
                all_notes.append(p.midi)
            elif isinstance(p, chord.Chord):
                all_durations.append(p.duration.quarterLength)
                for n in p.pitches:
                    all_notes.append(n.midi)
            elif isinstance(p, note.Rest):
                all_durations.append(p.duration.quarterLength)
        print(str(i))

    dtypes = np.unique([i for i in all_durations])
    enc = dict(zip(dtypes, list(range(0, len(dtypes)))))
    dec = {i: c for c, i in enc.items()}
    nu = len(enc)
    high = max(all_notes)
    low = min(all_notes)

    # Process each song and turn notes into many-hot vectors
    pitch = [[] for _ in all_songs]
    duration = [[] for _ in all_songs]

    for i, s in enumerate(all_songs) :
        # Get melody, duration, and offset for each song
        for p in s:
            if isinstance(p, note.Note):
                pitch[i].append(read_note(p.pitch))
                duration[i].append(int_to_vec(enc[p.duration.quarterLength], nu))
            elif isinstance(p, chord.Chord):
                pitch[i].append(read_chord(p))
                duration[i].append(int_to_vec(enc[p.duration.quarterLength], nu))
            elif isinstance(p, note.Rest):
                pitch[i].append(read_rest(p))
                duration[i].append(int_to_vec(enc[p.duration.quarterLength], nu))

        for j, v in enumerate(pitch[i]):
            pitch[i][j] = reduce(v, high, low)
        print("Finished {}".format(file))

    try:
        os.mkdir('data')
    except OSError:
        pass

    with open('data/param.pkl', 'wb') as f:
        pickle.dump((low, dec), f)
    with open('data/pitch.pkl', 'wb') as f:
        pickle.dump(pitch, f)
    with open('data/duration.pkl', 'wb') as f:
        pickle.dump(duration, f)


if __name__ == "__main__":
    start_time = time.time()

    read_midi('data/midi')

    print("Finished in {}".format(time.time() - start_time))
