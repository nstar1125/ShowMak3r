import os

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1].split('.')[0]) for fname in os.listdir(folder) if fname.startswith('iteration')]
    return max(saved_iters)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"