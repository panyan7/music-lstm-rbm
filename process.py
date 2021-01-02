import os
import time
import torch
import music21
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
    all_songs, all_keys, all_durations, all_notes = [], [], [], []
    # Get all songs in list all_songs
    for o in os.listdir(path):
        file = os.path.join(path, o)
        print("Parsing {}".format(file))
        s = music21.converter.parse(file)
        cs = s.chordify()
        all_songs.append(cs)

    # Preprocess all_songs to get the encodings for all durations and the corresponding table.
    # Also the highest and lowest pitch
    for i, song in enumerate(all_songs):
        all_keys.append(str(song.analyze('key')))
        for p in song:
            if isinstance(p, music21.note.Note):
                all_durations.append(p.duration.quarterLength)
                all_notes.append(p.midi)
            elif isinstance(p, music21.chord.Chord):
                all_durations.append(p.duration.quarterLength)
                for n in p.pitches:
                    all_notes.append(n.midi)
            elif isinstance(p, music21.note.Rest):
                all_durations.append(p.duration.quarterLength)

    dtypes = np.unique([i for i in all_durations])
    enc = dict(zip(dtypes, list(range(0, len(dtypes)))))
    dec = {i: c for c, i in enc.items()}
    nu = len(enc)
    high, low = max(all_notes), min(all_notes)

    # Process each song and turn notes into many-hot vectors
    pitch = []
    duration = []

    for i, s in enumerate(all_songs) :
        # Get melody, duration, and offset for each song
        v, u = [], []
        for p in s:
            if isinstance(p, music21.note.Note):
                v.append(reduce(read_note(p.pitch), high, low))
                u.append(int_to_vec(enc[p.duration.quarterLength], nu))
            elif isinstance(p, music21.chord.Chord):
                v.append(reduce(read_chord(p), high, low))
                u.append(int_to_vec(enc[p.duration.quarterLength], nu))
            elif isinstance(p, music21.note.Rest):
                v.append(reduce(read_rest(p), high, low))
                u.append(int_to_vec(enc[p.duration.quarterLength], nu))
        pitch.append(torch.stack(v))
        duration.append(torch.stack(u))
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
    with open('data/data.pkl', 'wb') as f:
        pickle.dump((pitch, duration), f)


start_time = time.time()
read_midi('data/midi')
print("Finished in {:.5f} seconds".format(time.time() - start_time))
