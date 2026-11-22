"""Trace a source compiler invocation and every descendant, one file per PID."""
from pathlib import Path
import subprocess


def traced(argv, prefix, timeout=240):
    prefix=Path(prefix)
    if list(prefix.parent.glob(prefix.name+'.*')): raise ValueError('trace output exists')
    p=subprocess.run(['strace','-ff','-qq','-s','4096','-e','trace=execve,execveat','-o',str(prefix),*argv],
                     stdout=subprocess.PIPE,check=True,timeout=timeout)
    logs=list(prefix.parent.glob(prefix.name+'.*'))
    if not logs or not all(f.suffix[1:].isdigit() for f in logs): raise ValueError('missing per-PID traces')
    trace='\n'.join(f.read_text() for f in logs)
    if str(argv[0]) not in trace or 'heir-opt' in trace or 'heir-translate' in trace:
        raise ValueError('missing native executable/forbidden subprocess')
    return p.stdout
