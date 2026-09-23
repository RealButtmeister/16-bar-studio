"""Run the unchanged velocity shaper inside the standalone application's runtime."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import runpy
import sys
import traceback


VELOCITY_WORKER_FLAG = '--eightbar-velocity-worker'


def maybe_run_velocity_worker():
    """Handle the private worker command before the GUI is imported or created."""
    if len(sys.argv) < 2 or sys.argv[1] != VELOCITY_WORKER_FLAG:
        return
    if len(sys.argv) != 5:
        if sys.stderr is not None:
            print('Velocity worker requires input MIDI, output MIDI, and report paths.',
                  file=sys.stderr)
        raise SystemExit(2)

    raw, final, report = (Path(value) for value in sys.argv[2:])
    # Never truncate the input or output MIDI when opening the text report.
    if report.resolve() in (raw.resolve(), final.resolve()):
        raise SystemExit(2)
    script = Path(__file__).with_name('velocity_shaper_original.py')
    original_args = sys.argv
    exit_code = 0
    try:
        # A windowed executable has no stdout/stderr. The report is an explicit
        # UTF-8 channel for both normal shaper output and useful failure details.
        with report.open('w', encoding='utf-8') as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    sys.argv = [str(script), str(raw), str(final)]
                    runpy.run_path(str(script), run_name='__main__')
                except SystemExit as exc:
                    if isinstance(exc.code, int):
                        exit_code = exc.code
                    elif exc.code is not None:
                        print(exc.code, file=sys.stderr)
                        exit_code = 1
                except Exception:
                    traceback.print_exc()
                    exit_code = 1
                finally:
                    sys.argv = original_args
    except OSError:
        if sys.stderr is not None:
            traceback.print_exc()
        exit_code = 1
    raise SystemExit(exit_code)
