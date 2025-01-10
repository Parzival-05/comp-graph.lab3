from typing import TypeAlias

from nptyping import Int, NDArray, Shape

RiceImageType: TypeAlias = NDArray[Shape["250, 250, 3"], Int]
