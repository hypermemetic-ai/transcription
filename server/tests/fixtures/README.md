# Test audio

`walk.s16` — 13.9 s of synthetic speech, 16 kHz mono s16le, no header. Made
on this machine with espeak-ng 1.52 (a formant synthesizer; no recording of a
person is in it, so there is no speaker's license to respect) and ffmpeg:

```sh
espeak-ng -v en-us -s 150 -g 12 -w walk.wav "So the plan for today is the kiln. [[ ]] We load the shelves in the morning. [[ ]] Then we fire them at noon and let them cool overnight."
ffmpeg -i walk.wav -af apad=pad_dur=0.5 -ar 16000 -ac 1 -f s16le walk.s16
```

`walk.txt` is the text it says.
