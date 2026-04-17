import os, imageio_ffmpeg, librosa, traceback
os.environ['PATH'] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
try:
    librosa.load('uploads/635eb1e6-f64f-4846-87c2-8d2ddb95eaaa.wav')
    print("Success")
except Exception as e:
    print(traceback.format_exc())
