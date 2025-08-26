from pylsl import StreamInlet, resolve_byprop

# Resolve EEG stream
streams = resolve_byprop('type', 'EEG', timeout=5.0)

if not streams:
    print("No EEG stream found")
else:
    info = streams[0]  # StreamInfo object
    inlet = StreamInlet(info)

    # Print basic stream metadata
    print("Stream name:", info.name())
    print("Stream type:", info.type())
    print("Channel count:", info.channel_count())
    print("Sampling rate:", info.nominal_srate())
    print("Manufacturer:", info.source_id())

    # Get XML description to extract channel labels (if available)
    desc = info.desc()
    ch = desc.child('channels').child('channel')
    print(info.as_xml())
    print("\nChannel labels:")
    channel_labels = []
    for i in range(info.channel_count()):
        label = ch.child_value('label')
        channel_labels.append(label)
        print(f"Channel {i}: {label}")
        ch = ch.next_sibling()

    # Start pulling samples
    for i in range(5):
        sample, ts = inlet.pull_sample(timeout=5.0)
        print(ts, sample)
