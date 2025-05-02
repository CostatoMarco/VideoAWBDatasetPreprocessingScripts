# MotionCam MCRAW decoder

A simple library for decoding files recorded by [MotionCam Pro](https://www.motioncamapp.com/).

## Usage

Look in `example.cpp` for a simple example on how to extract the RAW frames into DNGs and the audio into a WAV file.

To build the example:

```
mkdir build

cd build

cmake ..

make
```

To extract all frames from all videos inside a directory, run:

`./example <path to directory>`

## Sample Files

You can download a sample file from [here](https://storage.googleapis.com/motioncamapp.com/samples/007-VIDEO_24mm-240328_141729.0.mcraw).

## MotionCam Pro

MotionCam Pro is an app for Android that provides the ability to record RAW videos. Get it from the [Play Store](https://play.google.com/store/apps/details?id=com.motioncam.pro&hl=en&gl=US).

## THE EXAMPLE.CPP FILE WAS MODIFIED FOR THE PURPOUSE OF THIS PROJECT!!!!
