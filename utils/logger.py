"""Write one tab-separated metric table per fit phase.

Input is a sequence of named numeric rows. Output is ``progress.txt`` for
training and validation or ``test.txt`` for the final test evaluation.
Existing metric tables are never replaced.
"""

import os
import os.path as osp
import atexit

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """Write one tabular metric file without replacing prior results."""
    def __init__(self, output_dir: str,
                 output_fname: str = 'progress.txt') -> None:
        """Prepare a metric writer without creating an empty table.

        Parameters
        ----------
        output_dir : str
            Absolute fit directory.
        output_fname : str, optional
            Metric file name, default 'progress.txt'.
        """
        if output_dir is None:
            raise ValueError('Logger output_dir must be absolute.')
        if not osp.isabs(output_dir):
            raise ValueError(
                f'Logger output_dir must be absolute: {output_dir}'
            )
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_path = osp.join(self.output_dir, output_fname)
        self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers,\
                "Trying to introduce a new key {} that you didn't include in the first iteration".format(key)
        assert key not in self.log_current_row,\
            "You already set {} this iteration. Maybe you forgot to call dump_tabular()".format(key)
        self.log_current_row[key] = val

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if not self.log_headers:
            raise ValueError('No metrics are available to write.')
        if self.output_file is None:
            self.output_file = open(self.output_path, 'x')
            atexit.register(self.output_file.close)
            print(colorize(
                f'Logging data to {self.output_path}', 'green', bold=True
            ))
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)

        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()

        self.log_current_row.clear()
        self.first_row = False
