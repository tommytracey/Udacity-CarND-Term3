# Create video from image predictions
print('Creating video clip...')
clip = mpy.ImageSequenceClip('runs/1524868008.9052386', fps=15)
my_clip.write_videofile("video/video_out.mp4",fps=15, audio=False)
print('Video clip completed.')
