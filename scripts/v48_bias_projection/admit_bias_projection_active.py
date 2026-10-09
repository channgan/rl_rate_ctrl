"""Use the unchanged fixed-point admission harness with the v2 solver only."""
import admit_bias_projection as harness
from bias_projection_active import project

if __name__=='__main__':
    harness.project=project
    harness.main()
