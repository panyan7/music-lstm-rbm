import os
import time
import torch
import pickle
import argparse
import music21
from model import MusicGenerationModel


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


def main():
    starttime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--sample_step", type=int, default=10,
                        help="Number of rounds for Gibbs sampling")
    parser.add_argument("-l", "--length", type=int, default=100,
                        help="Length of the music")
    parser.add_argument("-t", "--tempo", type=float, default=75.0,
                        help="Tempo of the music")
    parser.add_argument("-f", "--filename", type=str, default="music.mid",
                        help="Output filename (.mid or .xml)")
    parser.add_argument("--data_path", type=str, default="data",
                        help="Path to saved data")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device")
    args = parser.parse_args()

    model = torch.load(args.data_path + '/model.pth')
    model = model.to(args.device)
    with open(args.data_path + '/data.pkl', 'rb') as f:
        (pitch, duration) = pickle.load(f)
    with open(args.data_path + '/param.pkl', 'rb') as f:
        (low, dec, dis_v, dis_u) = pickle.load(f)

    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    v0 = torch.bernoulli(dis_v).to(args.device)
    u0 = torch.zeros(model.nu).to(args.device)
    u0[torch.multinomial(dis_u, 1)[0]] = 1
    pitch_out, duration_out, prob = model.generate(v0, u0, args.length, args.sample_step)
    pitch_out, duration_out = pitch_out.to('cpu'), duration_out.to('cpu')

    turn_midi(pitch_out, duration_out, low, dec, args.filename, args.tempo)

    print("Finished in {:.5f}".format(time.time() - starttime))


if __name__ == '__main__':
    main()

