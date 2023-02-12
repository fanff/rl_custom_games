
from ale_py import ALEInterface

ale = ALEInterface()

from ale_py.roms import Breakout
ale.loadROM(Breakout)