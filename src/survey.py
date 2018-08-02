import common.data_utils
import common.bot

def create_question(bot_port, do_record_screen=True, do_generate_tts=False, shuffle_order=False):
  import learning.model
  import common.watson

  # Write question data
  video_order = [0, 1, 2, 3]
  if shuffle_order:
    random.shuffle(video_order)

  clips = [common.data_utils.get_random_clip() for _ in range(3)]
  subtitle = ''
  subtitles = []
  with jsonlines.open('questions.jsonl', mode='a') as writer:
    question = dict(
        ids=[],
        subtitle='',
        angles_expected=[],
        classes_expected=[],
        n_frames_expected=[],
        video_order=video_order)

    for clip in clips:
      question['ids'] += [clip['id']]
      subtitle += clip['subtitle'] + ' '
      subtitles += [str(clip['subtitle'])]
      question['angles_expected'] += clip['angles']
      question['classes_expected'] += [clip['class']]
      question['n_frames_expected'] += [len(clip['angles'])]

    question['subtitle'] = subtitle
    writer.write(question)

  print(subtitles)
  question_id = '.'.join(question['ids'])


  if do_generate_tts:
    # Generate TTS audio clip
    file_names_speech = []
    for i, sub in enumerate(subtitles):
      file_name_speech = os.path.join(common.data_utils.DEFAULT_TTS_PATH, question['ids'][i] + '.wav')
      common.watson.write_tts_clip(file_name_speech, subtitle)
      file_names_speech += [file_name_speech]

  # Record clips
  # Define some functions
  from subprocess import Popen
  from common.bot import BotController
  bot = common.bot.BotController(address=args.bot_address, port=args.bot_port)
  bot.leds_off()

  def record_screen(tag, time):
    if not do_record_screen:
      return None, None
    file_name = os.path.join(common.data_utils.DEFAULT_VIDEO_PATH, clip['id'] + '--' + tag + '.mp4')
    p_recording = Popen([
        'ffmpeg',
        '-framerate', '25',
        '-f', 'x11grab',
        '-s', '560x752',
        '-i', ':1+1280,270',
        '-t', str(time),
        str(file_name)
    ])

    return p_recording, file_name

  def record_pose_animation(frames, tag):
    time = int(math.ceil(len(frames) / 25)) + 1  # Extra second for margin
    proc, file_name = record_screen(tag, time)
    bot.play_angles(frames)

    if proc is not None:
      output = proc.wait()
      print(output)

    return file_name

  def countdown(count=3, sleep_time=1):
    print()
    bot.leds_off()
    while count > 0:
      print('Counting down... {}'.format(count))
      time.sleep(sleep_time)
      count -= 1
    print('BOOOM')
    bot.leds_on()

  # Now, use them
  bot.reset_pose()
  print('Playing expected')
  bot.leds_on()
  file_name_expected = record_pose_animation(question['angles_expected'], 'expected')
  bot.leds_off()

  bot.reset_pose()
  clusters = common.data_utils.get_clusters()
  predictions = learning.model.predict_classes(subtitles)
  animation = list(itertools.chain.from_iterable(clusters[x] for x in predictions))
  print('Playing clustered')
  bot.leds_on()
  file_name_cluster = record_pose_animation(animation, 'cluster')
  bot.leds_off()

  bot.reset_pose()
  animation = list(itertools.chain.from_iterable(learning.model.predict_sequences(subtitles)))
  print('Playing predicted')
  bot.leds_on()
  file_name_predicted = record_pose_animation(animation, 'predicted')
  bot.leds_off()

  bot.reset_pose()
  time_expected = int(math.ceil(len(question['angles_expected']) / 25)) + 1
  print('Playing native')
  bot.leds_on()
  proc, file_name_nao = record_screen('nao', time_expected)
  bot.say(subtitle)
  bot.leds_off()
  if proc is not None:
    proc.wait()

  def text_filter(text, start, end):
    text = text.replace('\'', '\\\'')
    return (';[v]drawtext=text=\'' + text + '\'' +
            ":enable='between(t,"+str(int(math.ceil(start)))+","+str(int(math.floor(end)))+")'"
            ":fontfile=DejaVuSans\\\:style=Bold" +
            ":x=(main_w/2-text_w/2)" +
            ":y=(main_h-(text_h*2)):fontsize=48:fontcolor=white" +
            ":borderw=2[v]")

  # Merge it all
  text_filters = ''
  start = 0
  for i, sub in enumerate(subtitles):
    end = start + float(question['n_frames_expected'][i]) / 25
    print(start)
    print(end)
    text_filters += text_filter(sub, start, end)
    start = end
    print(start)
    print(end)

  print(text_filters)

  subtitles = ''.join(text_filter(sub, question['n_frames_expected'][i], end) for i, sub in enumerate(subtitles))
  print(subtitles)

  if do_record_screen and do_generate_tts:
    video_file_names = [file_name_expected, file_name_cluster, file_name_predicted, file_name_nao]
    proc = Popen([
        'ffmpeg',
        '-i', video_file_names[video_order[0]],
        '-i', video_file_names[video_order[1]],
        '-i', video_file_names[video_order[2]],
        '-i', video_file_names[video_order[3]],
        '-i', file_names_speech[0], '-itsoffset', str(math.ceil(question['n_frames_expected'][0] / 25)),
        '-i', file_names_speech[1], '-itsoffset', str(math.ceil((question['n_frames_expected'][0] + question['n_frames_expected'][1]) / 25)),
        '-i', file_names_speech[2],
        '-filter_complex', ('[0:v][1:v][2:v][3:v]hstack=inputs=4[v]' +
                            text_filters +
                            ';[4:a][5:a][6:a]amix=3[a]'),
        '-map', '[v]',
        '-map', '[a]',
        # '-map', '4:a:0',
        os.path.join(common.data_utils.DEFAULT_VIDEO_PATH, clip['id'] + '--merged.mp4')
    ])
    proc.wait()
    print("Recorded. Saved merged video file.")
