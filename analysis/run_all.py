import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ['suite.py', 'cost.py', 'figures.py']:
    print(f'=== {script}')
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True, cwd=HERE)
print('done')
