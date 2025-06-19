from pylsl import StreamInlet, resolve_byprop

streams = resolve_byprop('type', 'EEG')
if not streams:
    print("No EEG stream found")
else:
    inlet = StreamInlet(streams[0])
    for i in range(5):
        print("stream found waiting for sample")
        sample, ts = inlet.pull_sample(5.0)
        print(ts, sample)