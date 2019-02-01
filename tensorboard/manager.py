# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Private utilities for managing multiple TensorBoard processes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import collections
import errno
import json
import os
import subprocess
import tempfile
import time

from tensorboard.util import tb_logging


# Information about a running TensorBoard instance.
TensorboardInfo = collections.namedtuple(
    "TensorboardInfo",
    (
        "version",  # tensorboard.version.VERSION
        "start_time",  # int: seconds since epoch (`int(time.time())`)
        "pid",
        "port",
        "path_prefix",
        "logdir",
        "db",
        "cache_key",  # opaque string
    ),
)


def _get_info_dir():
  """Get path to directory in which to store info files.

  The directory will be created if it does not exist.
  """
  path = os.path.join(tempfile.gettempdir(), ".tensorboard-info")
  if not os.path.exists(path):
    os.mkdir(path)
  return path


def _info_file_path():
  """Get path to info file for the current process."""
  return os.path.join(_get_info_dir(), "pid-%d.info" % os.getpid())


def write_info_file(tensorboard_info):
  """Write TensorboardInfo to the current process's info file.

  This should be called by `main` once the server is ready. When the
  server shuts down, `remove_info_file` should be called.
  """
  payload = json.dumps(tensorboard_info._asdict(), sort_keys=True, indent=4)
  with open(_info_file_path(), "w") as outfile:
    outfile.write("%s\n" % payload)


def remove_info_file():
  """Remove the current process's TensorboardInfo file, if it exists.

  If the file does not exist, no action is taken and no error is raised.
  """
  try:
    os.unlink(_info_file_path())
  except OSError as e:
    if e.errno == errno.ENOENT:
      # The user may have wiped their temporary directory or something.
      # Not a problem: we're already in the state that we want to be in.
      pass
    else:
      raise


def cache_key(working_directory, arguments):
  """Compute a `TensorboardInfo.cache_key` field.

  Args:
    working_directory: The directory from which TensorBoard was launched
      and relative to which paths like `--logdir` and `--db` are
      resolved.
    arguments: The command-line args to TensorBoard: `sys.argv[1:]`.
      Should be a list (or tuple), not an unparsed string. If you have a
      raw shell command, use `shlex.split` before passing it to this
      function.

  Returns:
    A string such that if two (prospective or actual) TensorBoard
    invocations have the same cache key then it is safe to use one in
    place of the other.
  """
  if not isinstance(arguments, (list, tuple)):
    raise TypeError(
        "'arguments' should be a list of arguments, but found: %r "
        "(use `shlex.split` if given a string)"
        % (arguments,)
    )
  datum = {"working_directory": working_directory, "arguments": arguments}
  return base64.b64encode(
      json.dumps(datum, sort_keys=True, separators=(",", ":"))
  )


def get_all():
  """Read TensorboardInfo values for running TensorBoard processes.

  This function may not provide a perfect snapshot of the set of running
  processes. Its result set may be incomplete if the user has cleaned
  their /tmp/ directory while TensorBoard processes are running. It may
  contain extraneous entries if TensorBoard processes exited uncleanly
  (e.g., with SIGKILL).

  Returns:
    A list of `TensorboardInfo` objects.
  """

  info_dir = _get_info_dir()
  results = []
  for filename in os.listdir(info_dir):
    filepath = os.path.join(info_dir, filename)
    with open(os.path.join(info_dir, filepath)) as infile:
      contents = infile.read()
    try:
      info = TensorboardInfo(**json.loads(contents))
    except Exception:
      tb_logging.get_logger().warning(
          "invalid info file: %r",
          filepath,
          exc_info=True,
      )
    else:
      results.append(info)
  return results


def _find_matching_instance(cache_key):
  infos = get_all()
  candidates = [info for info in infos if info.cache_key == cache_key]
  for candidate in sorted(candidates, key=lambda x: x.port):
    # TODO(@wchargin): Check here that the provided port is still live?
    return candidate
  return None



# Indicates that a call to `start` was compatible with an existing
# TensorBoard process, which can be reused according to the provided
# info.
StartReused = collections.namedtuple("StartReused", ("info",))

# Indicates that a call to `start` successfully launched a new
# TensorBoard process, which is available with the provided info.
StartLaunched = collections.namedtuple("StartLaunched", ("info",))

# Indicates that a call to `start` tried to launch a new TensorBoard
# instance, but the subprocess exited with the given exit code and
# output streams.
StartFailed = collections.namedtuple(
    "StartFailed",
    (
        "exit_code",  # int, as `Popen.returncode` (negative for signal)
        "stdout",
        "stderr",
    ),
)

# Indicates that a call to `start` launched a TensorBoard process, but
# that process neither exited nor wrote its info file within the allowed
# timeout period. The process may still be running under the included
# pid.
StartTimedOut = collections.namedtuple("StartTimedOut", ("pid",))


def start(arguments, timeout_seconds=10):
  match = _find_matching_instance(cache_key(os.getcwd(), arguments))
  if match:
    return StartReused(info=match)

  (stdout_fd, stdout_path) = tempfile.mkstemp(prefix="tensorboard-stdout-")
  (stderr_fd, stderr_path) = tempfile.mkstemp(prefix="tensorboard-stderr-")
  start_time = time.time()
  try:
    p = subprocess.Popen(
        ["tensorboard"] + arguments,
        stdout=stdout_fd,
        stderr=stderr_fd,
    )
  finally:
    os.close(stdout_fd)
    os.close(stderr_fd)

  poll_interval_seconds = 0.5
  end_time = start_time + timeout_seconds
  while time.time() < end_time:
    time.sleep(poll_interval_seconds)
    subprocess_result = p.poll()
    if subprocess_result is not None:
      with open(stdout_path) as stdout_infile:
        stdout = stdout_infile.read()
      with open(stderr_path) as stderr_infile:
        stderr = stderr_infile.read()
      return StartFailed(
          exit_code=subprocess_result,
          stdout=stdout,
          stderr=stderr,
      )
    for info in get_all():
      if info.pid == p.pid and info.start_time >= start_time:
        return StartLaunched(info=info)
  else:
    return StartTimedOut(pid=p.pid)
