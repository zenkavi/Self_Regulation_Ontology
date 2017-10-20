"""
Script to execute dimensional structure scripts
"""
import subprocess
EFA_cmd = 'python exploratory_fa.py'
hierarchical_cmd = 'python hierarchical_analysis.py'

commands = [EFA_cmd, hierarchical_cmd]
for cmd in commands:
    subprocess.call(cmd, shell=True)