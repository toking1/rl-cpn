from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_adaptation import main

if __name__ == "__main__":
    main('extension')
