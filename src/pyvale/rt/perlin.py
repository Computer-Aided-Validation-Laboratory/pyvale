import numpy as np
import numpy.typing as npt
import math

class Perlin:
    def __init__(self, point_count: int = 256):
        self.point_count = point_count

        randvec = np.random.uniform(-1, 1, size=(point_count, 3))
        norms = np.linalg.norm(randvec, axis=1, keepdims=True)
        # Avoid divide/by/zero
        norms[norms == 0] = 1.0
        self.randvec = randvec / norms

        self.perm_x = self._perlin_generate_perm()
        self.perm_y = self._perlin_generate_perm()
        self.perm_z = self._perlin_generate_perm()
    
    def noise(self, p) -> float:
        u = p[0] - math.floor(p[0])
        v = p[1] - math.floor(p[1])
        w = p[2] - math.floor(p[2])

        i = int(math.floor(p[0]))
        j = int(math.floor(p[1]))
        k = int(math.floor(p[2]))

        c = np.empty((2, 2, 2, 3))

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    c[di][dj][dk] = self.randvec[
                        self.perm_x[(i + di) & 255] ^
                        self.perm_y[(j + dj) & 255] ^
                        self.perm_z[(k + dk) & 255]
                    ]
        
        return self._perlin_interp(c, u, v, w)


    def turb(self, p, depth: int) -> float:
        accum = 0.0
        temp_p = p
        weight = 1.0

        for i in range(depth):
            accum += weight * self.noise(temp_p)
            weight *= 0.5
            temp_p *= 2
        
        return math.fabs(accum)

    def _perlin_generate_perm(self) -> npt.NDArray[np.int_]:
        p = np.arange(self.point_count, dtype=int)
        
        np.random.shuffle(p)
        return p

    # def permute(p: npt.NDArray[np.int_], n: int) -> None:
    #     for i in range(n - 1, 0, -1):
    #         target = np.random.randint(0, i + 1)  # randint is inclusive on both ends in C++
    #         p[i], p[target] = p[target], p[i]
    def _perlin_interp(self, c, u: float, v: float, w: float) -> float:
        uu = u*u*(3-2*u)
        vv = v*v*(3-2*v)
        ww = w*w*(3-2*w)

        accum = 0.0

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight_v = np.array([u-i, v-j, w-k])
                    accum += (i*uu + (1-i)*(1-uu)) \
                           * (j*vv + (1-j)*(1-vv)) \
                           * (k*ww + (1-k)*(1-ww)) \
                           * np.dot(c[i][j][k], weight_v)

        return accum