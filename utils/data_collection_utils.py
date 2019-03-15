import os
import numpy as np

class ImageQueue(object):
  """An image queue holding each stream's most recent image.

  Basically implements a process-safe collections.deque(maxlen=1).
  """

  def __init__(self):
    self.lock = multiprocessing.Lock()
    self._queue = multiprocessing.Queue(maxsize=1)

  def append(self, data):
    with self.lock:
      if self._queue.full():
        # Pop the first element.
        _ = self._queue.get()
      self._queue.put(data)

  def get(self):
    with self.lock:
      return self._queue.get()

  def empty(self):
    return self._queue.empty()

  def close(self):
    return self._queue.close()
             
def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

def setup_paths(args):
  """Sets up the necessary paths to collect videos."""
  assert args.dataset
  assert args.num_views
  assert args.expdir


  # Setup directory for final images used to create videos for this sequence.
  tmp_imagedir = os.path.join(args.tmp_imagedir, args.dataset, )
  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)

  # Create a base directory to hold all sequence videos if it doesn't exist.
  vidbase = os.path.join(args.expdir, args.dataset, args.viddir, )
  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if args.debug_vids:
    debug_base = os.path.join('%s_debug' % args.viddir, args.dataset,
                              )
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debug_path = '%s/%s.mp4' % (debug_base, seqname)

  return view_dirs, vid_paths, debug_path, seqname

def setup_paths_w_depth(args):
  """Sets up the necessary paths to collect videos."""
  assert args.dataset
  assert args.num_views
  assert args.expdir

  # Setup directory for final images used to create videos for this sequence.
  tmp_imagedir = os.path.join(args.tmp_imagedir, args.dataset)
  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)
  tmp_depthdir = os.path.join(args.tmp_imagedir,  args.dataset, 'depth', )
  if not os.path.exists(tmp_depthdir):
    os.makedirs(tmp_depthdir)
  # Create a base directory to hold all sequence videos if it doesn't exist.
  vidbase = os.path.join(args.expdir, args.dataset, args.viddir, )
  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

    # Setup depth directory
  depthbase = os.path.join(args.expdir, args.dataset, args.depthdir, )
  if not os.path.exists(depthbase):
    os.makedirs(depthbase)
  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir, args)
  view_dirs_depth = get_view_dirs_depth(vidbase, tmp_depthdir, args)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)
  depth_paths = []
  for idx, _ in enumerate(view_dirs_depth):
    depth_path = os.path.join(depthbase, '%s_view%d.mp4' % (seqname, idx))
    depth_paths.append(depth_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if args.debug_vids:
    debug_base = os.path.join(args.expdir, args.dataset, '%s_debug' % args.viddir, 
                              )

    debug_path = '%s/%s.mp4' % (debug_base, seqname)
    debug_path_depth = '%s/%s_depth.mp4' % (debug_base, seqname)

  return view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth

def get_view_dirs(vidbase, tmp_imagedir, args):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if args.seqname:
    seqname = args.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(vidbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(vidbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs = [os.path.join(
      tmp_imagedir, '%s_view%d' % (seqname, v)) for v in range(args.num_views)]
  for d in view_dirs:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs, seqname


def get_view_dirs_depth(depthbase, tmp_depthdir, args):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if args.seqname:
    seqname = args.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(depthbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(depthbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs_depth = [os.path.join(
      tmp_depthdir, '%s_view%d' % (seqname, v)) for v in range(args.num_views)]
  for d in view_dirs_depth:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs_depth